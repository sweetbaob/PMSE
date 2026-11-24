"""Checked DLinear integration; other Proceed backbones do not enter this module.

Native DLinear remains unnormalized. Use --normalization RevIN explicitly for a
separately named experiment, with the same normalization in both stages.
"""
from pathlib import Path

import torch

VERSION = 'dlinear-proceed-v1'


def enabled(args):
    return getattr(args, 'model', None) == 'DLinear'


def configure(args):
    if not enabled(args):
        return
    if args.compile or args.use_multi_gpu or args.local_rank != -1:
        raise ValueError('Checked DLinear currently supports single-device, uncompiled runs.')
    if args.online_method not in (None, 'Online', 'Proceed', 'OneNet'):
        raise ValueError('Unsupported DLinear online method.')
    if args.online_method == 'OneNet' and args.normalization:
        raise ValueError('DLinear + OneNet uses the native DLinear input path.')
    if args.leakage or args.ema != 0:
        raise ValueError('Checked DLinear requires causal feedback and ema=0.')
    if args.online_method == 'Proceed' and args.tune_mode != 'down_up':
        raise ValueError('Checked DLinear + Proceed currently supports down_up adapters.')
    if args.normalization not in (None, 'RevIN'):
        raise ValueError('DLinear normalization must be omitted or explicitly RevIN.')
    if args.lradj == 'TST':
        raise ValueError('Use a DLinear schedule such as type3 or constant, not PatchTST OneCycle.')
    for name in ['learning_rate', 'backbone_learning_rate', 'pretrain_learning_rate', 'online_learning_rate']:
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            raise ValueError(f'{name} must be positive.')
    args.save_opt = True
    args.model_id += '_dlinear_v1' + ('_individual' if args.individual else '_shared')
    print('DLinear interface:', VERSION,
          '| normalization:', args.normalization or 'none (native DLinear; --revin is PatchTST-specific)',
          '| patience:', args.patience, '| max epochs:', args.train_epochs)
    if args.diff_online_lr:
        print('DLinear: --diff_online_lr has no implementation in this archive; '
              'use --backbone_learning_rate for the explicit offline backbone rate. '
              'Online FT uses --online_learning_rate for all live groups.')


def pretrain_lr(args):
    if args.pretrain_learning_rate is not None:
        return args.pretrain_learning_rate
    import settings
    key = args.model + ('_RevIN' if args.normalization else '')
    table = settings.pretrain_lr_online_dict if args.online_method else settings.pretrain_lr_dict
    try:
        return table[key][args.dataset]
    except KeyError as exc:
        raise ValueError('Specify --pretrain_learning_rate to match the DLinear pretraining run.') from exc


def wrapped(model):
    return hasattr(model, 'generator') and hasattr(model, 'recent_batch')


def assert_optimizer(model, optimizer):
    live = {id(p): n for n, p in model.named_parameters()}
    enrolled = [id(p) for group in optimizer.param_groups for p in group['params']]
    if len(enrolled) != len(set(enrolled)):
        raise RuntimeError('DLinear optimizer contains duplicate parameters.')
    if set(enrolled) != set(live):
        stale = len(set(enrolled) - set(live))
        missing = [live[k] for k in set(live) - set(enrolled)]
        raise RuntimeError(f'DLinear optimizer ownership mismatch: stale={stale}, missing={missing}')


def optimizer(exp, filter_frozen=True, return_self=True, model=None):
    target = exp.model if model is None else model
    if return_self and exp.model_optim is not None:
        assert_optimizer(target, exp.model_optim)
        return exp.model_optim
    args = exp.args
    backbone = target.backbone if wrapped(target) else target
    backbone_params = list(backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    # Enroll currently frozen parameters too: Proceed's validation/warmup toggles
    # requires_grad. Autograd still decides which enrolled parameters update.
    base_lr = (args.backbone_learning_rate if wrapped(target)
               and args.backbone_learning_rate is not None else args.learning_rate)
    groups = [dict(params=backbone_params, lr=base_lr, name='dlinear_backbone',
                   dlinear_lr_scale=base_lr / args.learning_rate)]
    other = [p for p in target.parameters() if id(p) not in backbone_ids]
    if other:
        groups.append(dict(params=other, lr=args.learning_rate, name='proceed_adapter',
                           dlinear_lr_scale=1.0))
    result = getattr(torch.optim, args.optim)(groups, lr=args.learning_rate)
    assert_optimizer(target, result)
    if return_self:
        exp.model_optim = result
    return result


def signature(model, args):
    result = dict(format=VERSION, stage='proceed' if wrapped(model) else 'pretrain',
                  dataset=args.dataset, seed=getattr(args, 'run_seed', None),
                  seq_len=args.seq_len, pred_len=args.pred_len, channels=args.enc_in,
                  individual=args.individual, normalization=args.normalization,
                  affine=bool(args.affine) if args.normalization else None,
                  subtract_last=bool(args.subtract_last) if args.normalization else None,
                  parameter_shapes={n: list(p.shape) for n, p in model.named_parameters()},
                  warmup='validation_labels_end_before_test; native two-phase Proceed warmup')
    if wrapped(model):
        result['adaptation'] = dict(tune_mode=args.tune_mode, concept_dim=args.concept_dim,
                                    bottleneck_dim=args.bottleneck_dim,
                                    individual_generator=args.individual_generator,
                                    freeze=args.freeze)
    return result


def group_names(model, opt):
    assert_optimizer(model, opt)
    names = {id(p): n for n, p in model.named_parameters()}
    return [[names[id(p)] for p in group['params']] for group in opt.param_groups]


def checkpoint_state(state, model, args, opt):
    if 'model' not in state or 'model_optim' not in state:
        raise RuntimeError('DLinear checkpoint requires both model and optimizer state.')
    # Exp_Basic omits frozen weights in some modes; this checked interface
    # deliberately saves a full, strict model so pretraining can be recovered.
    state['model'] = model.state_dict()
    state['_dlinear_interface'] = signature(model, args)
    optimized_model = model.backbone if args.online_method == 'OneNet' else model
    state['_dlinear_optimizer_names'] = group_names(optimized_model, opt)
    state['_dlinear_requires_grad'] = {n: p.requires_grad for n, p in model.named_parameters()}
    if wrapped(model):
        state['_dlinear_recent_batch'] = model.recent_batch.detach().clone()
    return state


def load_pretrained(path, model, args, device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing matching DLinear pretraining checkpoint: {path}')
    state = torch.load(path, map_location=device)
    if state.get('_dlinear_interface') != signature(model, args):
        raise RuntimeError('DLinear pretraining metadata/normalization/seed mismatch. '
                           'Retrain with the checked interface; old runs are preserved separately.')
    model.load_state_dict(state['model'], strict=True)
    # Proceed replaces Linear Parameter objects. Import weights only, then build
    # a fresh optimizer on the final wrapped model. Never retain dead Adam refs.
    print('Loaded matching DLinear weights; building a fresh optimizer for this stage.')
    return model


def restore(exp, state, model=None):
    target = exp.model if model is None else model
    if state.get('_dlinear_interface') != signature(target, exp.args):
        raise RuntimeError('DLinear checkpoint interface mismatch; refusing permissive loading.')
    optimized_model = target.backbone if exp.args.online_method == 'OneNet' else target
    if state.get('_dlinear_optimizer_names') != group_names(optimized_model, exp.model_optim):
        raise RuntimeError('DLinear checkpoint optimizer group names/order do not match.')
    target.load_state_dict(state['model'], strict=True)
    exp.model_optim.load_state_dict(state['model_optim'])
    if wrapped(target):
        context = state.get('_dlinear_recent_batch')
        if context is None:
            raise RuntimeError('DLinear + Proceed checkpoint is missing the concept reference.')
        target.recent_batch = context.detach().clone().to(next(target.parameters()).device)
    for name, param in target.named_parameters():
        param.requires_grad_(state['_dlinear_requires_grad'][name])
    assert_optimizer(optimized_model, exp.model_optim)
    return target
