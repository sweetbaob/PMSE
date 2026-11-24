"""Opt-in checkpoint/optimizer interfaces; no changes to Proceed's E/E'/G logic."""
import copy
import torch


def enabled(args):
    return (getattr(args, 'backbone_variant', 'original') == 'jointr_nor'
            or getattr(args, 'checked_backbone_io', False))


def validate_options(args):
    if not enabled(args):
        return
    if args.model != 'PatchTST' or args.online_method not in (None, 'Proceed'):
        raise ValueError('The checked JointR-noR comparison supports PatchTST + Proceed/pretraining only')
    if args.freeze or args.compile or args.use_multi_gpu or args.local_rank != -1:
        raise ValueError('Checked backbone integration currently requires single-device, uncompiled full tuning')
    if not args.save_opt:
        raise ValueError('Checked backbone checkpoints require --save_opt')
    if args.leakage or args.ema != 0:
        raise ValueError('This integration is scoped to the audited non-leakage, ema=0 configuration')
    if args.online_method and args.tune_mode != 'down_up':
        raise ValueError('This integration currently supports the audited down_up adaptation mode')


def live_optimizer(model, optimizer):
    """Retain existing live Adam state, remove replaced objects, cover all live params.

    add_adapters_ replaces Linear Parameter objects. Preserve state only for
    unchanged live objects (e.g. BN/position), never transfer a dead object's
    Adam moments by merely matching tensor shape. No optimizer reset/LR change.
    Currently frozen parameters are enrolled because original validation later
    unfreezes the backbone; requires_grad still controls each original phase.
    """
    live = set(model.parameters())
    for group in optimizer.param_groups:
        group['params'] = [p for p in group['params'] if p in live]
    for p in list(optimizer.state):
        if p not in live:
            del optimizer.state[p]
    registered = [p for g in optimizer.param_groups for p in g['params']]
    if len(registered) != len(set(registered)):
        raise RuntimeError('Duplicate live parameters in the optimizer')
    missing = [p for p in model.parameters() if p not in set(registered)]
    if missing:
        optimizer.add_param_group({'params': missing})
    if set(p for g in optimizer.param_groups for p in g['params']) != live:
        raise RuntimeError('Optimizer does not cover the live backbone and Proceed parameters')


def manifest(model, args, optimizer):
    wrapped = hasattr(model, 'generator') and hasattr(model, 'backbone')
    names = {p: name for name, p in model.named_parameters()}
    groups = []
    if optimizer is not None:
        for group in optimizer.param_groups:
            if any(p not in names for p in group['params']):
                raise RuntimeError('A stale optimizer reference must not be saved/loaded')
            groups.append([names[p] for p in group['params']])
    result = dict(format='checked-proceed-backbone-v1',
                  variant=getattr(args, 'backbone_variant', 'original'),
                  stage='proceed' if wrapped else 'pretrain',
                  dataset=args.dataset, seed=getattr(args, 'run_seed', None),
                  seq_len=args.seq_len, pred_len=args.pred_len, channels=args.enc_in,
                  offline_reconstruction_weight=.5 if getattr(args, 'backbone_variant', 'original') == 'jointr_nor' else 0.,
                  online_reconstruction_weight=0., stage1_forecast_weight=0.,
                  optimizer_ownership='all-live-objects; retain-only-unchanged-Adam-state',
                  optimizer_parameter_names=groups,
                  parameter_shapes={name: list(p.shape) for name, p in model.named_parameters()})
    if wrapped:
        result['adaptation'] = dict(tune_mode=args.tune_mode, concept_dim=args.concept_dim,
                                    bottleneck_dim=args.bottleneck_dim,
                                    individual_generator=args.individual_generator,
                                    dim_name_dict={k: list(v) for k, v in model.generator.dim_name_dict.items()})
    return result


def checkpoint_state(state, model, args, optimizer):
    if 'model' not in state or 'model_optim' not in state:
        raise RuntimeError('Checked backbone save requires full model + optimizer state')
    state = copy.copy(state)
    state['_backbone_interface'] = manifest(model, args, optimizer)
    if hasattr(model, 'recent_batch'):
        state['_proceed_recent_batch'] = model.recent_batch.detach().cpu().clone()
    return state


def check_checkpoint(state, model, args, optimizer):
    expected = manifest(model, args, optimizer)
    if state.get('_backbone_interface') != expected:
        raise RuntimeError('Incompatible backbone/G/optimizer checkpoint. Train this variant in its own '
                           'namespace; original or simplified JointR weights are not implicitly migrated.')
    if 'model' not in state or 'model_optim' not in state:
        raise RuntimeError('Checked checkpoint is missing model/optimizer state')
    context = state.get('_proceed_recent_batch')
    if hasattr(model, 'recent_batch'):
        if not isinstance(context, torch.Tensor) or context.ndim != 3 or context.shape[-2:] != (args.seq_len + args.pred_len, args.enc_in):
            raise RuntimeError('Missing or misaligned nonpersistent Proceed recent_batch')
    return {k: state[k] for k in ('model', 'model_optim')}, context


def restore_context(model, context):
    if hasattr(model, 'recent_batch'):
        model.recent_batch = context.detach().to(next(model.parameters()).device).clone()
