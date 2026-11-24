"""Thin integration around the archived Proceed experiment classes.

Algorithms remain in vendor/onlinetsf. This module owns chronology, strict state
restoration, checkpoint identities, and the otherwise missing offline execution.
"""
import copy
import gc
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .config import ROOT, read, digest
from .data import Windows, recent, loader
from pmse.utils import save_json, run_lock, seed_all


def namespace(cfg, data, device, checkpoint_root, pretraining=False, model_name=None):
    args = read(ROOT / "configs/baseline_defaults.json")
    args.update(cfg["model"])
    settings = cfg["pretraining"] if pretraining else cfg["baseline"]
    method = cfg["method"]
    native = model_name or (
        "FSNet"
        if method == "FSNet"
        else ("FSNet_Ensemble" if method == "OneNet" and cfg["backbone"] == "TCN" else cfg["backbone"])
    )
    args.update(
        model=native,
        dataset=cfg["dataset"],
        enc_in=data.C,
        c_out=data.C,
        dec_in=data.C,
        label_len=0,
        timeenc=cfg["baseline"]["timeenc"],
        freq=cfg["frequency"],
        normalization=cfg["baseline"]["normalization"],
        online_method=None if pretraining else method,
        use_gpu=str(device).startswith("cuda"),
        gpu=0,
        use_multi_gpu=False,
        pin_gpu=True,
        local_rank=-1,
        num_workers=0,
        optim=cfg["baseline"]["optimizer"],
        weight_decay=0.01 if cfg["baseline"]["optimizer"] == "AdamW" else 0.0,
        batch_size=settings["batch_size"],
        benchmark_pretraining=pretraining,
        pretraining_micro_batch=min(
            settings["batch_size"], settings.get("micro_batch", settings["batch_size"])
        )
        if native in ("PatchTST", "iTransformer")
        else settings["batch_size"],
        train_epochs=settings["epochs"],
        patience=settings["patience"],
        learning_rate=cfg["recipe"]["pretrain_lr"] if pretraining else cfg["recipe"]["offline_lr"],
        online_learning_rate=cfg["recipe"]["online_lr"],
        concept_dim=cfg["recipe"].get("concept_dim", 200),
        bottleneck_dim=cfg["recipe"].get("bottleneck_dim", 48),
        checkpoints=str(checkpoint_root),
        save_opt=True,
        pretrain=False,
        whole_model=True,
        continual=True,
        strict_extension_warmup=True,
        backbone_variant="original",
        compile=False,
        lradj="type3",
        wo_clip=cfg["baseline"].get("wo_clip", False),
        borders=[[0, data.borders[0] - data.L, data.borders[1] - data.L], list(data.borders)],
    )
    if pretraining and cfg["backbone"] == "iTransformer":
        args["timeenc"] = 1
    if native.startswith("FSNet"):
        args["normalization"] = "RevIN"
        args["timeenc"] = 2
    if cfg.get("smoke"):
        args.update(test_train_num=64, selected_data_num=2)
    return SimpleNamespace(**args)


class CommonExperiment:
    """MRO mixin: all optimizer references point to the final live model."""

    def _acquire_device(self):
        return torch.device("cuda:0" if self.args.use_gpu else "cpu")

    def _select_optimizer(self, filter_frozen=True, return_self=True, model=None):
        if return_self and self.model_optim is not None:
            return self.model_optim
        target = model if model is not None else self.model
        # OneNet owns two further optimizers for its mixture and decision network.
        if self.args.online_method == "OneNet" and hasattr(target, "decision"):
            target = target.backbone
        # Proceed changes requires_grad between phases. Keep live frozen params in
        # the optimizer; Adam skips parameters with no gradient in that phase.
        optimizer = getattr(torch.optim, self.args.optim)(target.parameters(), lr=self.args.learning_rate)
        if return_self:
            self.model_optim = optimizer
        return optimizer

    def _get_data(self, flag, **kwargs):
        if flag in getattr(self, "online_phases", []):
            ds = recent(self.benchmark_data, flag, self.args.timeenc)
            return ds, loader(ds)
        ds = Windows(self.benchmark_data, self.benchmark_data.origins[flag], self.args.timeenc)
        batch_size = self.args.batch_size
        if flag == "val" and self.args.benchmark_pretraining:
            batch_size = min(batch_size, self.args.pretraining_micro_batch)
        return ds, loader(ds, batch_size, flag == "train")

    def _update(self, batch, criterion, optimizer, scaler=None):
        micro = self.args.pretraining_micro_batch
        n = len(batch[0])
        if not self.args.benchmark_pretraining or n <= micro:
            return super()._update(batch, criterion, optimizer, scaler)
        if scaler is not None or isinstance(optimizer, tuple):
            raise ValueError("Pretraining accumulation expects one FP32 optimizer")
        optimizer.zero_grad(set_to_none=True)
        total = batch[0].new_zeros(())
        outputs = []
        for start in range(0, n, micro):
            chunk = [x[start : start + micro] for x in batch]
            pred = self.forward(chunk)
            loss = self.train_loss(criterion, chunk, pred)
            weight = len(chunk[0]) / n
            (loss * weight).backward()
            total += loss.detach() * weight
            outputs.append(pred.detach())
        optimizer.step()
        return total, torch.cat(outputs)

    def vali(self, valid_data, valid_loader, criterion):
        if not self.args.benchmark_pretraining:
            return super().vali(valid_data, valid_loader, criterion)
        self.phase = "val"
        self.model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in valid_loader:
                output = self.forward(batch)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                error = output.double() - batch[1].double()
                total += error.square().sum().item()
                count += error.numel()
        return total / count

    def state_dict(self, *args, **kwargs):
        context = {}
        for name, module in self.model.named_modules():
            extra = {}
            for key in (
                "recent_batch",
                "q_ema",
                "trigger",
                "try_trigger",
                "flag_update",
                "flag_current",
                "flag_online_learning",
            ):
                if hasattr(module, key):
                    value = getattr(module, key)
                    extra[key] = value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
            if extra:
                context[name] = extra
        payload = {
            "model": self.model.state_dict(),
            "model_optim": self.model_optim.state_dict(),
            "context": context,
            "requires_grad": {n: p.requires_grad for n, p in self.model.named_parameters()},
        }
        for name in ("opt_w", "opt_bias"):
            if hasattr(self, name):
                payload[name] = getattr(self, name).state_dict()
        if hasattr(self, "bias"):
            payload["online_bias"] = self.bias.detach().clone()
        return payload

    def load_state_dict(self, state_dict, model=None, strict=True):
        target = self.model if model is None else model
        target.load_state_dict(state_dict["model"], strict=True)
        self.model_optim.load_state_dict(state_dict["model_optim"])
        for name, values in state_dict.get("context", {}).items():
            module = target.get_submodule(name)
            for key, value in values.items():
                setattr(
                    module, key, value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
                )
        # An absent q_ema must remain absent on restoring an earlier FSNet state.
        for name, module in target.named_modules():
            if hasattr(module, "q_ema") and "q_ema" not in state_dict.get("context", {}).get(name, {}):
                delattr(module, "q_ema")
        for name, p in target.named_parameters():
            p.requires_grad_(state_dict["requires_grad"][name])
        for name in ("opt_w", "opt_bias"):
            if name in state_dict:
                getattr(self, name).load_state_dict(state_dict[name])
        if "online_bias" in state_dict:
            self.bias = state_dict["online_bias"].detach().clone()
        return target


def create(cfg, data, device, checkpoint_root, pretraining=False, model_name=None):
    from exp import Exp_Main, Exp_Online, Exp_Proceed, Exp_ER, Exp_DERpp, Exp_FSNet, Exp_OneNet, Exp_SOLID

    mapping = {
        "Static": Exp_Main,
        "Online": Exp_Online,
        "Proceed": Exp_Proceed,
        "ER": Exp_ER,
        "DERpp": Exp_DERpp,
        "FSNet": Exp_FSNet,
        "OneNet": Exp_OneNet,
        "SOLID++": Exp_SOLID,
    }
    cls = (Exp_FSNet if model_name == "FSNet" else Exp_Main) if pretraining else mapping[cfg["method"]]
    combined = type("Benchmark" + cls.__name__, (CommonExperiment, cls), {})
    obj = combined(namespace(cfg, data, device, checkpoint_root, pretraining, model_name))
    obj.benchmark_data = data
    live = {id(p) for p in obj.model.parameters()}
    owned = [p for g in obj.model_optim.param_groups for p in g["params"]]
    if any(id(p) not in live for p in owned) or len({id(p) for p in owned}) != len(owned):
        raise RuntimeError("Optimizer holds stale or duplicate parameters")
    expected = {
        id(p)
        for p in (
            obj.model.backbone.parameters()
            if cfg["method"] == "OneNet" and not pretraining
            else obj.model.parameters()
        )
    }
    if {id(p) for p in owned} != expected:
        raise RuntimeError("Optimizer does not cover live parameters")
    return obj


def pretrained(cfg, data, seed, device, cache, code_id, model_name=None):
    """No historical checkpoint is guessed from a filename or another random seed."""
    pre_cfg = copy.deepcopy(cfg)
    if model_name == "FSNet":
        pre_cfg["model"]["seq_len"] = 60
        pre_cfg["baseline"].update(normalization="RevIN", timeenc=2, optimizer="AdamW")
        # This branch has its own documented LR; no backbone LR is substituted.
        rates = {"ETTh2": 0.003, "ETTm1": 0.001, "Weather": 0.003, "ECL": 0.003, "Traffic": 0.003}
        pre_cfg["recipe"]["pretrain_lr"] = rates.get(cfg["dataset"], 0.001)
        # Reuse source arrays and scaler, but change only the window length.
        data = copy.copy(data)
        data.L = 60
        tr, va, te = data.borders
        H = data.H
        data.origins = dict(
            train=np.arange(60, tr - H + 1), val=np.arange(tr, va - H + 1), test=np.arange(va, te - H + 1)
        )
    spec = dict(
        model=model_name or cfg["backbone"],
        dataset=cfg["dataset"],
        seed=seed,
        shape=pre_cfg["model"],
        settings=pre_cfg["pretraining"],
        normalization=pre_cfg["baseline"]["normalization"],
        optimizer=pre_cfg["baseline"]["optimizer"],
        lr=pre_cfg["recipe"]["pretrain_lr"],
        data=data.provenance,
        code=code_id,
    )
    identity = digest(spec)
    folder = Path(cache) / identity
    with run_lock(folder):
        checkpoint = folder / "best.pth"
        if (folder / "complete.json").exists():
            if not checkpoint.exists():
                raise RuntimeError(f"Broken cache {folder}")
            return torch.load(checkpoint, map_location=device, weights_only=False)
        if (folder / "spec.json").exists():
            raise RuntimeError(f"Partial pretraining cache {folder}; inspect it before resuming.")
        save_json(folder / "spec.json", spec)
        seed_all(seed)
        exp = create(pre_cfg, data, device, folder, True, model_name)
        exp.train("fit")
        state = copy.deepcopy(exp.model.state_dict())
        # q_ema is not a registered parameter in the archived FSNet code.
        context = {
            n: {
                k: getattr(m, k).detach().clone() if torch.is_tensor(getattr(m, k)) else getattr(m, k)
                for k in ("q_ema", "trigger", "try_trigger")
                if hasattr(m, k)
            }
            for n, m in exp.model.named_modules()
            if any(hasattr(m, k) for k in ("q_ema", "trigger"))
        }
        optim_by_name = {
            name: copy.deepcopy(exp.model_optim.state[p])
            for name, p in exp.model.named_parameters()
            if p in exp.model_optim.state
        }
        value = dict(model=state, context=context, optimizer_by_name=optim_by_name, identity=identity)
        torch.save(value, checkpoint)
        save_json(folder / "complete.json", {"identity": identity, "spec": spec})
        del exp
        gc.collect()
        return value


def restore_branch(target, value, optimizer=None):
    target.load_state_dict(value["model"], strict=True)
    for name, extra in value.get("context", {}).items():
        for key, v in extra.items():
            setattr(target.get_submodule(name), key, v)
    if optimizer is not None:
        for name, p in target.named_parameters():
            if name in value.get("optimizer_by_name", {}):
                optimizer.state[p] = copy.deepcopy(value["optimizer_by_name"][name])


def execute(cfg, data, seed, folder, device, cache, code_id):
    method = cfg["method"]
    folder = Path(folder)
    seed_all(seed)
    weights = None
    fsnet = None
    if method not in ("FSNet",) and not (method == "OneNet" and cfg["backbone"] == "TCN"):
        weights = pretrained(cfg, data, seed, device, cache, code_id)
    if method == "OneNet" and cfg["backbone"] != "TCN":
        fsnet = pretrained(cfg, data, seed, device, cache, code_id, model_name="FSNet")
    seed_all(seed)
    exp = create(cfg, data, device, folder / "checkpoints")
    if weights is not None:
        if method == "Proceed":
            target = exp.model.backbone
        elif method == "OneNet":
            target = exp.model.backbone.encoder_time
        else:
            target = exp.model
        # Proceed replaces Linear/Conv module types, while retaining these keys.
        restore_branch(target, weights, exp.model_optim)
    if fsnet is not None:
        restore_branch(exp.model.backbone.encoder, fsnet, exp.model_optim)
    if method in ("Proceed", "OneNet", "FSNet"):
        exp.train("fit")
    save_json(
        folder / "execution.json",
        dict(
            pretrain_identity=weights["identity"] if weights else None,
            fsnet_identity=fsnet["identity"] if fsnet else None,
            optimizer=exp.args.optim,
            epochs=exp.args.train_epochs,
        ),
    )
    for g in exp.model_optim.param_groups:
        g["lr"] = cfg["recipe"]["online_lr"]
    seed_all(seed)
    if method == "Static":
        mse, mae, *_ = exp.test("evaluation")
    elif method == "SOLID++":
        # Native SOLID selects samples within the latest fully matured history.
        extra = exp.test_train_num + data.H - 1
        origins = np.arange(data.origins["test"][0] - extra, data.origins["test"][-1] + 1)
        ds = Windows(data, origins, exp.args.timeenc)
        begin = int(origins[0]) - data.L
        end = data.borders[2]
        ds.data_x = ds.data_y = data.z[begin:end]
        ds.data_stamp = data.stamps(np.arange(begin, end), exp.args.timeenc)
        mse, mae, *_ = exp.online(online_data=ds)
    else:
        exp.update_valid(recent(data, "val", exp.args.timeenc))
        mse, mae, *_ = exp.online(online_data=recent(data, "test", exp.args.timeenc))
    if not np.isfinite([mse, mae]).all():
        raise FloatingPointError("Non-finite baseline metrics")
    return dict(
        mse=float(mse),
        mae=float(mae),
        n_test=len(data.origins["test"]),
        metric_scale="train_standardized",
        scope="full100",
        method=method,
        checks=dict(
            full_H_maturity=True,
            validation_targets_end=data.borders[1],
            test_targets_start=data.borders[1],
            optimizer_live=True,
        ),
    )
