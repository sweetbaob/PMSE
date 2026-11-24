"""Reproducible command-line experiment matrix and result collection."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
import itertools
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
import traceback
from .config import (
    ROOT,
    DATASETS,
    BACKBONES,
    METHODS,
    SEEDS,
    resolve,
    read,
    digest,
    implementation,
    software_versions,
)

# Legacy import names are confined to the vendored experiment engine.
sys.path.insert(0, str(ROOT / "vendor/onlinetsf"))


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, s):
        for f in self.files:
            f.write(s)
            f.flush()
        return len(s)

    def flush(self):
        for f in self.files:
            f.flush()

    def isatty(self):
        return False


def matrix(args):
    for backbone, ds, method in itertools.product(args.backbones, args.datasets, args.methods):
        if method == "FSNet" and backbone != "TCN":
            continue
        horizons = args.horizons or read(ROOT / "configs/datasets" / f"{ds}.json")["horizons"]
        for h, seed in itertools.product(horizons, args.seeds):
            yield resolve(
                ds,
                backbone,
                method,
                h,
                seed,
                dict(epochs=args.epochs, batch_size=args.batch_size, online_lr=args.online_lr),
            )


def run_one(cfg, data, folder, device, cache, code):
    from pmse.utils import save_json, run_lock

    folder = Path(folder)
    software = software_versions()
    identity = digest(dict(config=cfg, data=data.provenance, code=code, software=software))
    with run_lock(folder):
        completion = folder / "result.json"
        if completion.exists():
            result = read(completion)
            if result["fingerprint"] != identity:
                raise RuntimeError(
                    f"Completed output has different config/data/code: {folder}; choose another --run-name"
                )
            print(f"SKIP complete and identical: {folder}", flush=True)
            return result
        if (folder / "resolved.json").exists():
            raise RuntimeError(
                f"Incomplete output preserved at {folder}; inspect log and choose another --run-name. No silent restart."
            )
        save_json(
            folder / "resolved.json",
            dict(
                config=cfg,
                data=data.provenance,
                implementation=code,
                fingerprint=identity,
                runtime=dict(python=sys.version, device=str(device), software=software),
            ),
        )
        with (folder / "run.log").open("a", encoding="utf-8", buffering=1) as log, redirect_stdout(
            Tee(sys.stdout, log)
        ), redirect_stderr(Tee(sys.stderr, log)):
            start = time.perf_counter()
            try:
                if cfg["method"] == "PMSE":
                    from pmse.engine import run_seed

                    raw = run_seed(cfg, data, cfg["seed"], folder / "engine", device)
                    result = dict(
                        mse=raw["mse"]["PFTC"],
                        mae=raw["mae"]["PFTC"],
                        n_test=raw["n_test"],
                        metric_scale=raw["metric_scale"],
                        scope=raw["scope"],
                        variant="PFTC",
                        diagnostic_PFT=dict(mse=raw["mse"]["PFT"], mae=raw["mae"]["PFT"]),
                    )
                else:
                    from .baseline import execute

                    result = execute(
                        cfg,
                        data,
                        cfg["seed"],
                        folder,
                        device,
                        cache,
                        digest(dict(code=code, software=software)),
                    )
                result.update(
                    fingerprint=identity,
                    config_fingerprint=digest(cfg),
                    protocol=cfg["protocol_version"],
                    dataset=cfg["dataset"],
                    backbone=cfg["backbone"],
                    method=cfg["method"],
                    seed=cfg["seed"],
                    seq_len=data.L,
                    pred_len=data.H,
                    normalization_policy=cfg["normalization_policy"],
                    recipe_source=cfg["recipe_source"],
                    seconds=time.perf_counter() - start,
                )
                save_json(completion, result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
                return result
            except BaseException as exc:
                save_json(folder / "failure.json", dict(error=repr(exc), traceback=traceback.format_exc()))
                raise


def run(args):
    configs = list(matrix(args))
    if not configs:
        raise ValueError("No valid experiments in selection")
    if args.dry_run:
        rows = [
            {k: c[k] for k in ("dataset", "backbone", "method", "seed", "recipe_source")}
            | dict(L=c["model"]["seq_len"], H=c["model"]["pred_len"], recipe=c["recipe"])
            for c in configs
        ]
        if args.plan_file:
            plan = Path(args.plan_file)
            plan.parent.mkdir(parents=True, exist_ok=True)
            plan.write_text(
                json.dumps({"count": len(rows), "experiments": rows}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(
            json.dumps(
                {
                    "count": len(rows),
                    "experiments": rows if len(rows) <= 12 else rows[:12],
                    "preview_truncated": len(rows) > 12,
                    "plan_file": args.plan_file,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        raise ValueError("--run-name must be a simple folder name")
    if args.device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly")
    device = torch.device("cpu" if args.device == "cpu" else "cuda:0")
    from .data import BenchmarkData

    code = implementation()
    failures = []
    previous = None
    data = None
    for i, cfg in enumerate(configs, 1):
        print(
            f'[{i}/{len(configs)}] {cfg["backbone"]} {cfg["dataset"]} H{cfg["model"]["pred_len"]} {cfg["method"]} seed{cfg["seed"]}',
            flush=True,
        )
        key = (cfg["dataset"], cfg["backbone"], cfg["model"]["seq_len"], cfg["model"]["pred_len"])
        folder = (
            Path(args.output_root)
            / args.run_name
            / cfg["backbone"]
            / cfg["dataset"]
            / f'H{cfg["model"]["pred_len"]}'
            / cfg["method"]
            / f'seed{cfg["seed"]}'
        )
        try:
            if key != previous:
                data = None
                data = BenchmarkData(cfg, args.data_root, device)
                previous = key
            run_one(cfg, data, folder, device, Path(args.output_root) / "_pretrain", code)
        except Exception as exc:
            failures.append(dict(config=cfg, error=repr(exc)))
            traceback.print_exc()
            if not args.keep_going:
                raise
    if failures:
        from pmse.utils import save_json

        save_json(Path(args.output_root) / args.run_name / "failures.json", failures)
        raise RuntimeError(f"{len(failures)} experiment(s) failed; inspect failures.json")


def check_data(args):
    from .data import BenchmarkData
    from pmse.utils import save_json

    records = []
    for ds in args.datasets:
        cfg = resolve(ds, "PatchTST", "PMSE", 48, 2021)
        d = BenchmarkData(cfg, args.data_root, "cpu")
        p = d.provenance
        p["dataset"] = ds
        p["target_intervals_half_open"] = [
            [0, d.borders[0]],
            [d.borders[0], d.borders[1]],
            [d.borders[1], d.borders[2]],
        ]
        records.append(p)
        print(
            json.dumps(
                {k: p[k] for k in ("dataset", "rows", "channels", "borders", "origins", "calendar")},
                ensure_ascii=False,
            ),
            flush=True,
        )
    save_json(args.report, dict(protocol="proceed_code_fixed_ett_v1", datasets=records))
    print(f"Data audit saved: {args.report}")


def collect(args):
    records = []
    groups = {}
    for file in sorted(Path(args.input).rglob("result.json")):
        r = read(file)
        if r.get("scope") != "full100":
            continue
        row = {
            k: r[k]
            for k in (
                "backbone",
                "dataset",
                "pred_len",
                "seq_len",
                "method",
                "seed",
                "mse",
                "mae",
                "protocol",
            )
        }
        row["result_path"] = str(file.resolve())
        records.append(row)
        # Identity includes every config field except the random seed.
        resolved = read(file.parent / "resolved.json")
        cfg = resolved["config"].copy()
        cfg.pop("seed", None)
        key = (
            r["backbone"],
            r["dataset"],
            r["pred_len"],
            r["seq_len"],
            r["method"],
            digest(
                dict(
                    config=cfg,
                    data=resolved["data"],
                    code=resolved["implementation"],
                    software=resolved["runtime"].get("software", {}),
                )
            ),
        )
        groups.setdefault(key, []).append(r)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    def write_csv(path, rows, keys):
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(
        out / "runs.csv",
        records,
        [
            "backbone",
            "dataset",
            "pred_len",
            "seq_len",
            "method",
            "seed",
            "mse",
            "mae",
            "protocol",
            "result_path",
        ],
    )
    summaries = []
    for key, rs in groups.items():
        by_seed = {}
        for r in rs:
            if r["seed"] in by_seed:
                raise RuntimeError(f'Duplicate result for seed {r["seed"]}: {key}; narrow --input')
            by_seed[r["seed"]] = r
        complete = set(by_seed) == set(SEEDS)
        row = dict(zip(("backbone", "dataset", "pred_len", "seq_len", "method", "group_fingerprint"), key))
        row.update(
            n_seeds=len(by_seed),
            seeds=";".join(map(str, sorted(by_seed))),
            complete_three_seeds=complete,
            mse_mean=statistics.mean(r["mse"] for r in rs),
            mae_mean=statistics.mean(r["mae"] for r in rs),
            mse_std=statistics.stdev(r["mse"] for r in rs) if len(rs) > 1 else "",
            mae_std=statistics.stdev(r["mae"] for r in rs) if len(rs) > 1 else "",
        )
        summaries.append(row)
    keys = [
        "backbone",
        "dataset",
        "pred_len",
        "seq_len",
        "method",
        "group_fingerprint",
        "n_seeds",
        "seeds",
        "complete_three_seeds",
        "mse_mean",
        "mse_std",
        "mae_mean",
        "mae_std",
    ]
    write_csv(out / "summary.csv", summaries, keys)
    print(
        json.dumps(
            dict(
                runs=len(records),
                groups=len(summaries),
                complete_groups=sum(r["complete_three_seeds"] for r in summaries),
                output=str(out),
            ),
            ensure_ascii=False,
        )
    )


def smoke(args):
    """Small synthetic integration test; results are forbidden from the main table."""
    import numpy as np
    import torch
    from .data import BenchmarkData
    from pmse.utils import save_json

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    report = []
    code = implementation()
    root = Path(args.output)
    rng = np.random.default_rng(18)
    values = np.sin(np.arange(228)[:, None] / 8 + np.arange(3)[None, :]) + rng.normal(0, 0.03, (228, 3))
    for backbone in args.backbones:
        for method in args.methods:
            if method == "FSNet" and backbone != "TCN":
                continue
            cfg = resolve("PEMS03", backbone, method, 12, 2021, {"epochs": 1, "batch_size": 8})
            cfg["model"].update(
                seq_len=60 if backbone == "TCN" else 64,
                d_model=8,
                n_heads=2,
                e_layers=1,
                d_ff=16,
                tcn_output_dims=16,
                tcn_hidden_dims=8,
                tcn_depth=1,
            )
            cfg["data"]["split"] = {"type": "fixed", "boundaries": [180, 204, 228]}
            cfg["memory"].update(short_period=6, long_period=12, width=8)
            cfg["training"].update(new_lrs=[1e-4], micro_batch=8)
            cfg["statistics"]["bootstrap_draws"] = 20
            cfg["smoke"] = True
            data = BenchmarkData(cfg, device=device, values=values)
            folder = root / backbone / method
            result = run_one(cfg, data, folder, device, root / "_pretrain", code)
            result["scope"] = "synthetic_smoke_not_paper"
            save_json(folder / "result.json", result)
            report.append(
                dict(backbone=backbone, method=method, mse=result["mse"], mae=result["mae"], passed=True)
            )
    save_json(
        root / "verification.json", dict(kind="synthetic_integration_not_main_experiment", checks=report)
    )


def main():
    p = argparse.ArgumentParser(
        description="PMSE benchmark: four backbones, eight datasets, full-H delayed feedback"
    )
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="Execute a serial matrix. Use separate GPU jobs for parallelism.")
    r.add_argument("--methods", nargs="+", choices=METHODS, default=["PMSE", "Proceed"])
    r.add_argument("--backbones", nargs="+", choices=BACKBONES, default=list(BACKBONES))
    r.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    r.add_argument("--horizons", nargs="+", type=int)
    r.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    r.add_argument("--gpu", default="0", help="Physical CUDA GPU index, one GPU per process")
    r.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--data-root", default=os.environ.get("PMSE_DATA_ROOT", str(ROOT / "data")))
    r.add_argument("--output-root", default=os.environ.get("PMSE_OUTPUT_ROOT", str(ROOT / "outputs")))
    r.add_argument("--run-name", default="main_v1")
    r.add_argument("--epochs", type=int, help="Explicit override recorded in every output")
    r.add_argument("--batch-size", type=int)
    r.add_argument("--online-lr", type=float)
    r.add_argument("--keep-going", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--plan-file")
    r.set_defaults(func=run)
    c = sub.add_parser("check-data")
    c.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    c.add_argument("--data-root", default=os.environ.get("PMSE_DATA_ROOT", str(ROOT / "data")))
    c.add_argument("--report", default=str(ROOT / "verification/data_audit.json"))
    c.set_defaults(func=check_data)
    c = sub.add_parser("collect")
    c.add_argument("--input", default=str(ROOT / "outputs/main_v1"))
    c.add_argument("--output", default=str(ROOT / "outputs/tables"))
    c.set_defaults(func=collect)
    c = sub.add_parser("smoke")
    c.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    c.add_argument("--backbones", nargs="+", choices=BACKBONES, default=list(BACKBONES))
    c.add_argument("--threads", type=int, default=2)
    c.add_argument("--output", default=str(ROOT / "verification/smoke"))
    c.set_defaults(func=smoke)
    args = p.parse_args()
    for key in ("epochs", "batch_size", "threads"):
        if hasattr(args, key) and getattr(args, key) is not None and getattr(args, key) < 1:
            p.error(f'--{key.replace("_","-")} must be positive')
    if hasattr(args, "seeds") and any(s not in SEEDS for s in args.seeds):
        p.error("The main protocol uses seeds 2021, 2022, 2023")
    args.func(args)


if __name__ == "__main__":
    main()
