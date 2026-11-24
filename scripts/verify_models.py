"""Real-data forward/backward checks. Never trains or emits a paper result."""
import argparse
import gc
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/onlinetsf")]
from pmse_benchmark.config import resolve, DATASETS, BACKBONES
from pmse_benchmark.data import BenchmarkData, Windows, loader
from pmse_benchmark.baseline import create
from pmse.engine import build_model, forward
from pmse.utils import save_json, seed_all
from pmse.data import read_series


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--report", default=str(ROOT / "verification/model_shapes.json"))
    args = p.parse_args()
    torch.set_num_threads(2)
    rows = []
    for ds in DATASETS:
        base = resolve(ds, "PatchTST", "PMSE", 48, 2021)
        raw, columns, sources = read_series(base, args.data_root)
        original = BenchmarkData(base, args.data_root, "cpu")
        for bb in BACKBONES:
            for h in base["horizons"]:
                cfg = resolve(ds, bb, "PMSE", h, 2021)
                data = BenchmarkData(cfg, device="cpu", values=raw)
                data.marks1, data.marks2 = original.marks1, original.marks2
                t = int(data.origins["train"][0])
                seed_all(2021)
                model = build_model(cfg, data.C, "cpu")
                model.train()
                out = forward(model, data, [t])
                target = data.target([t])
                if out.shape != target.shape:
                    raise AssertionError((ds, bb, h, out.shape, target.shape))
                loss = (out - target).square().mean()
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads):
                    raise AssertionError("PMSE gradient failure")
                del model, out, grads
                gc.collect()
                bc = resolve(ds, bb, "Proceed", h, 2021)
                seed_all(2021)
                exp = create(bc, data, "cpu", ROOT / "verification/unused_checkpoints")
                batch = next(iter(loader(Windows(data, [t], exp.args.timeenc))))
                loss, out = exp._update(batch, exp._select_criterion(), exp.model_optim)
                if tuple(out.shape) != (1, h, data.C) or not torch.isfinite(loss):
                    raise AssertionError("Proceed shape/loss failure")
                rows.append(
                    dict(
                        dataset=ds,
                        backbone=bb,
                        L=data.L,
                        H=h,
                        C=data.C,
                        pmse_forward_backward=True,
                        proceed_forward_backward=True,
                    )
                )
                print("PASS", ds, bb, "L", data.L, "H", h, flush=True)
                del exp, data, batch, out
                gc.collect()
        del original, raw
        gc.collect()
    save_json(args.report, dict(kind="real_data_one_training_window_not_main_experiments", cases=rows))


if __name__ == "__main__":
    main()
