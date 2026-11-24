"""Resolve every experiment before execution; no hidden model-specific CLI overrides."""
import copy
import hashlib
import json
import importlib.metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("ETTh2", "ETTm1", "Weather", "ECL", "Traffic", "ExchangeRate", "Solar", "PEMS03")
BACKBONES = ("PatchTST", "iTransformer", "TCN", "DLinear")
METHODS = ("PMSE", "Proceed", "Static", "Online", "ER", "DERpp", "SOLID++", "OneNet", "FSNet")
SEEDS = (2021, 2022, 2023)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def implementation():
    return {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ("src", "vendor", "configs")
        for p in sorted((ROOT / folder).rglob("*"))
        if p.is_file() and p.suffix in (".py", ".json")
    }


def software_versions():
    """Library versions participate in result and shared-checkpoint identities."""
    return {
        name: importlib.metadata.version(name)
        for name in ("torch", "numpy", "pandas", "scikit-learn", "einops", "transformers")
    }


def resolve(dataset, backbone, method, horizon, seed, overrides=None):
    if dataset not in DATASETS or backbone not in BACKBONES or method not in METHODS:
        raise ValueError(f"Unsupported combination: {dataset}/{backbone}/{method}")
    if method == "FSNet" and backbone != "TCN":
        raise ValueError("FSNet is the TCN-family baseline; use --backbones TCN.")
    cfg = read(ROOT / "configs/datasets" / f"{dataset}.json")
    if horizon not in cfg["horizons"]:
        raise ValueError(f'{dataset}: H={horizon} is not in {cfg["horizons"]}')
    preset = read(ROOT / "configs/backbones" / f"{backbone}.json")
    cfg["model"].update(preset.get("model", {}))
    cfg["model"].update(preset.get("dataset_model_overrides", {}).get(dataset, {}))
    cfg["model"].update(seq_len=cfg["lengths"][backbone], pred_len=horizon)
    cfg.update(
        backbone=backbone,
        method=method,
        seed=int(seed),
        model_kind="pmse",
        protocol_version="proceed_code_fixed_ett_v1",
        normalization_policy="inherited_native_v1",
    )
    recipes = read(ROOT / "configs/recipes.json")[backbone][dataset]
    entry = recipes["methods"].get(method, recipes["methods"].get("Online", {}))
    selected = entry.get("by_horizon", {}).get(str(horizon))
    fallback = dict(
        offline_lr=recipes["pretrain_lr"], online_lr=cfg["online"]["lr"], concept_dim=200, bottleneck_dim=48
    )
    cfg["recipe"] = dict(fallback, **(selected or {}), pretrain_lr=recipes["pretrain_lr"])
    cfg["recipe_source"] = dict(
        source=entry.get("source", "declared_extension_default"),
        horizon_matched=selected is not None and "transferred_from_horizon" not in selected,
        status=(
            "transferred_horizon"
            if selected and "transferred_from_horizon" in selected
            else ("script" if selected else "declared_default_not_paper_tuned")
        ),
    )
    if method in ("ER", "DERpp", "Static"):
        cfg["recipe"]["offline_lr"] = recipes["pretrain_lr"]
    if method == "PMSE":
        cfg["recipe_source"] = {
            "source": f"configs/datasets/{dataset}.json",
            "status": "pmse_inherited_preset_not_baseline_grid",
        }
    cfg["baseline"] = dict(
        optimizer="AdamW" if backbone in ("TCN", "iTransformer") else "Adam",
        epochs=10 if backbone == "iTransformer" else (100 if backbone == "DLinear" else 25),
        patience=3,
        batch_size=16 if method == "Proceed" else 32,
        normalization="RevIN" if backbone == "TCN" else None,
        timeenc=1 if backbone == "iTransformer" else 2,
    )
    cfg["pretraining"] = dict(epochs=10 if backbone == "iTransformer" else 100, patience=3, batch_size=32)
    pre_script = recipes.get("pretraining_script", {})
    if "batch_size" in pre_script:
        cfg["pretraining"]["batch_size"] = int(pre_script["batch_size"])
    cfg["pretraining"]["micro_batch"] = min(cfg["pretraining"]["batch_size"], cfg["training"]["micro_batch"])
    if selected:
        cfg["baseline"]["epochs"] = min(
            int(selected.get("epochs", cfg["baseline"]["epochs"])), 25 if backbone != "DLinear" else 100
        )
        cfg["baseline"]["wo_clip"] = selected.get("wo_clip", False)
    if selected and "batch_size" in selected:
        cfg["baseline"]["batch_size"] = selected["batch_size"]
    if method == "OneNet" and backbone == "iTransformer":
        cfg["baseline"]["timeenc"] = "onenet_itransformer"
    if overrides:
        for key, value in overrides.items():
            if value is None:
                continue
            if key == "epochs":
                cfg["training"]["max_epochs"] = value
                cfg["baseline"]["epochs"] = cfg["pretraining"]["epochs"] = value
            elif key == "batch_size":
                cfg["training"]["batch_size"] = value
                cfg["training"]["micro_batch"] = min(value, cfg["training"]["micro_batch"])
                cfg["baseline"]["batch_size"] = cfg["pretraining"]["batch_size"] = value
                cfg["pretraining"]["micro_batch"] = min(value, cfg["pretraining"]["micro_batch"])
            elif key == "online_lr":
                cfg["online"]["lr"] = cfg["recipe"]["online_lr"] = value
            else:
                raise ValueError(f"Unknown override: {key}")
        cfg["explicit_overrides"] = {k: v for k, v in overrides.items() if v is not None}
    return copy.deepcopy(cfg)
