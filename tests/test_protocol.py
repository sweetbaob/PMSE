"""Protocol invariants: run with python -m unittest discover -s tests -v."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/onlinetsf")]
from pmse_benchmark.config import resolve, DATASETS, BACKBONES, METHODS
from pmse_benchmark.data import BenchmarkData, Windows, recent, loader
from pmse_benchmark.baseline import create


def small(backbone="PatchTST", method="Proceed"):
    cfg = resolve("PEMS03", backbone, method, 12, 2021, {"epochs": 1, "batch_size": 8})
    cfg["model"].update(
        seq_len=64,
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
    cfg["smoke"] = True
    return cfg


def equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.values = np.random.default_rng(77).normal(size=(228, 3))

    def test_declared_matrix(self):
        count = 0
        for ds in DATASETS:
            for bb in BACKBONES:
                hs = [12, 24, 48] if ds == "PEMS03" else [24, 48, 96]
                for method in METHODS:
                    if method == "FSNet" and bb != "TCN":
                        continue
                    for h in hs:
                        cfg = resolve(ds, bb, method, h, 2021)
                        self.assertEqual(cfg["model"]["pred_len"], h)
                        count += 1
        self.assertEqual(count, 792)

    def test_ett_boundaries_and_lengths(self):
        for ds, borders in [("ETTh2", [2880, 3600, 14400]), ("ETTm1", [11520, 14400, 57600])]:
            for bb in BACKBONES:
                c = resolve(ds, bb, "Proceed", 48, 2021)
                self.assertEqual(c["data"]["split"]["boundaries"], borders)
                self.assertEqual(borders[0] / borders[2], 0.2)
                self.assertEqual((borders[1] - borders[0]) / borders[2], 0.05)
                self.assertEqual((borders[2] - borders[1]) / borders[2], 0.75)
        for bb in ("PatchTST", "iTransformer", "DLinear"):
            self.assertEqual(resolve("ETTh2", bb, "PMSE", 48, 2021)["model"]["seq_len"], 336)
        self.assertEqual(resolve("ETTh2", "TCN", "PMSE", 48, 2021)["model"]["seq_len"], 60)

    def test_no_cross_boundary_targets(self):
        cfg = small()
        d = BenchmarkData(cfg, values=self.values)
        for flag, begin, end in [("train", 64, 180), ("val", 180, 204), ("test", 204, 228)]:
            origins = d.origins[flag]
            self.assertEqual(origins[0], begin)
            self.assertEqual(origins[-1] + d.H, end)
            if flag != "train":
                ds = recent(d, flag)
                self.assertEqual(len(ds), len(origins))
                recent_batch, current = ds[0]
                self.assertTrue(torch.equal(recent_batch[1], d.z[begin - d.H : begin]))
                self.assertTrue(torch.equal(current[1], d.z[begin : begin + d.H]))

    def test_future_does_not_change_scaler_or_past(self):
        cfg = small()
        d = BenchmarkData(cfg, values=self.values)
        changed = self.values.copy()
        changed[204:] += 10000
        other = BenchmarkData(cfg, values=changed)
        self.assertTrue(np.array_equal(d.scaler.mean_, other.scaler.mean_))
        self.assertTrue(torch.equal(d.window([204]), other.window([204])))
        for a, b in zip(d.memory([204]), other.memory([204])):
            self.assertTrue(torch.equal(a, b))

    def test_proceed_validation_restores_all_training_state(self):
        for bb in BACKBONES:
            cfg = small(bb)
            d = BenchmarkData(cfg, values=self.values)
            with tempfile.TemporaryDirectory() as folder:
                exp = create(cfg, d, "cpu", folder)
                batch = next(iter(loader(Windows(d, d.origins["train"], exp.args.timeenc), 8)))
                exp._update(batch, exp._select_criterion(), exp.model_optim)
                before = copy.deepcopy(exp.state_dict())
                valid = recent(d, "val", exp.args.timeenc)
                mse = exp.vali(valid, loader(valid), exp._select_criterion())
                self.assertTrue(np.isfinite(mse))
                self.assertTrue(equal(before, exp.state_dict()), bb)

    def test_fsnet_memory_trigger_is_portable_and_checkpointed(self):
        from layers.ts2vec.fsnet_ import SamePadConv

        layer = SamePadConv(4, 8, 3)
        layer.trigger = 1
        out = layer(torch.randn(2, 4, 17))
        self.assertEqual(tuple(out.shape), (2, 8, 17))
        out.square().mean().backward()
        layer.store_grad()
        state = copy.deepcopy(layer.state_dict())
        clone = SamePadConv(4, 8, 3)
        clone.load_state_dict(state, strict=True)
        self.assertTrue(torch.equal(layer.grads, clone.grads))
        self.assertTrue(torch.equal(layer.f_grads, clone.f_grads))

    def test_accumulation_preserves_global_batch_optimizer_step(self):
        cfg = small("iTransformer")
        cfg["model"]["dropout"] = 0.0
        cfg["pretraining"].update(batch_size=8, micro_batch=8)
        d = BenchmarkData(cfg, values=self.values)
        with tempfile.TemporaryDirectory() as folder:
            full = create(cfg, d, "cpu", folder, pretraining=True)
            cfg["pretraining"]["micro_batch"] = 2
            micro = create(cfg, d, "cpu", folder, pretraining=True)
            micro.load_state_dict(copy.deepcopy(full.state_dict()))
            batch = next(iter(loader(Windows(d, d.origins["train"], 1), 8)))
            full._update(batch, full._select_criterion(), full.model_optim)
            micro._update(batch, micro._select_criterion(), micro.model_optim)
            for a, b in zip(full.model.parameters(), micro.model.parameters()):
                self.assertTrue(torch.allclose(a, b, atol=2e-6, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
