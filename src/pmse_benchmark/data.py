"""A common chronology, scaler and origin schedule for both algorithm engines."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pmse.data import SeriesData


class BenchmarkData(SeriesData):
    def __init__(self, cfg, data_root=None, device="cpu", values=None):
        super().__init__(cfg, data_root, device, values)
        from util.timefeatures import time_features, time_features_from_frequency_str

        n = len(self.z)
        spec = cfg["data"]
        freq = cfg["frequency"]
        date_col = spec.get("date_column")
        observed = values is None and spec["format"] == "csv" and date_col
        if observed:
            dates = pd.read_csv(Path(data_root) / spec["path"], usecols=[date_col]).iloc[:n]
            dates = dates.rename(columns={date_col: "date"})
            clock = pd.to_datetime(dates["date"], errors="raise")
            # Keep observation order, including explicitly declared Weather clock anomalies.
            m1 = time_features(dates.copy(), timeenc=1, freq=freq)
            m2 = time_features(dates.copy(), timeenc=2, freq=freq)
        else:
            m1 = np.zeros((n, len(time_features_from_frequency_str(freq))), dtype=np.float32)
            m2 = np.zeros((n, 7), dtype=np.float32)
        self.marks1 = torch.as_tensor(m1.astype(np.float32), device=self.device)
        self.marks2 = torch.as_tensor(m2.astype(np.float32), device=self.device)
        self.provenance["calendar"] = dict(
            source="observed_csv" if observed else "zero_no_timestamps",
            frequency=freq,
            policy=cfg["calendar_policy"],
        )
        if observed:
            self.provenance["calendar"].update(
                monotonic_increasing=bool(clock.is_monotonic_increasing),
                duplicate_clock_rows=int(clock.duplicated().sum()),
                first_clock_label=str(clock.iloc[0]),
                last_clock_label=str(clock.iloc[-1]),
            )

    def stamps(self, indices, encoding=None):
        encoding = encoding if encoding is not None else (1 if self.cfg["backbone"] == "iTransformer" else 2)
        indices = torch.as_tensor(indices, device=self.device, dtype=torch.long)
        if encoding == "onenet_itransformer":
            return torch.cat((self.marks1[indices], self.marks2[indices]), dim=-1)
        return (self.marks1 if encoding == 1 else self.marks2)[indices]

    def marks(self, origins):
        t = torch.as_tensor(origins, device=self.device, dtype=torch.long).reshape(-1)
        return self.stamps(t[:, None] - self.L + torch.arange(self.L, device=self.device))


class Windows(Dataset):
    """Origins are absolute: x=[t-L,t), y=[t,t+H); no split boundary guessing."""

    def __init__(self, data, origins, encoding=2):
        self.data, self.origins, self.encoding = data, np.asarray(origins), encoding
        if not len(self.origins) or self.origins[0] < data.L:
            raise ValueError("Empty/invalid origin schedule")

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, i):
        t = int(self.origins[i])
        d = self.data
        return [
            d.z[t - d.L : t],
            d.z[t : t + d.H],
            d.stamps(np.arange(t - d.L, t), self.encoding),
            d.stamps(np.arange(t, t + d.H), self.encoding),
        ]


def recent(data, flag, encoding=2):
    from data_provider.data_loader import Dataset_Recent

    origins = data.origins[flag]
    return Dataset_Recent(
        Windows(data, np.arange(origins[0] - data.H, origins[-1] + 1), encoding), gap=data.H
    )


def loader(dataset, batch_size=1, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)
