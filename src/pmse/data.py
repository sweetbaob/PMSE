"""Train-only scaling, explicit online splits, causal phase-memory extraction."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from .utils import sha256

KEY, PAD = 16, 2048


def reconstruct_windowed_npz(folder):
    """Recover raw chronology only if all X/Y overlaps and split boundaries agree.

    The supplied train/val/test partitions are transport containers, not the online
    experiment split. No rescaling, averaging, imputation or timestamp invention.
    """
    pieces, tail, sources = [], None, []
    for split in ('train', 'val', 'test'):
        path = Path(folder) / f'{split}.npz'
        with np.load(path, allow_pickle=False) as archive:
            needed = {'x', 'y', 'x_offsets', 'y_offsets'}
            if not needed.issubset(archive.files):
                raise ValueError(f'{path}: missing {needed - set(archive.files)}')
            x, y = archive['x'], archive['y']
            if x.ndim != 4 or y.ndim != 4 or x.shape[0] != y.shape[0] or x.shape[2:] != y.shape[2:] or x.shape[-1] != 1:
                raise ValueError('Expected chronological X/Y [N,window,channels,1], one feature.')
            lx, ly = x.shape[1], y.shape[1]
            if not np.array_equal(archive['x_offsets'].ravel(), np.arange(1-lx, 1)):
                raise ValueError('Unsupported input offsets.')
            if not np.array_equal(archive['y_offsets'].ravel(), np.arange(1, ly+1)):
                raise ValueError('Unsupported target offsets.')
            series = np.concatenate([x[0, :, :, 0], x[1:, -1, :, 0], y[-1, :, :, 0]], axis=0)
            for s in range(0, len(x), 256):
                indices = np.arange(s, min(s+256, len(x)))[:, None]
                if not np.array_equal(x[s:s+256, :, :, 0], series[indices + np.arange(lx)]):
                    raise ValueError(f'{split}: X windows are shuffled, gapped or inconsistent.')
                if not np.array_equal(y[s:s+256, :, :, 0], series[indices + lx + np.arange(ly)]):
                    raise ValueError(f'{split}: X/Y scales or target overlaps disagree.')
            overlap = lx + ly - 1
            if tail is not None and not np.array_equal(tail, series[:overlap]):
                raise ValueError(f'{split}: train/validation/test boundary is discontinuous.')
            pieces.append(series if tail is None else series[overlap:])
            tail = series[-overlap:].copy()
            sources.append(path)
            del x, y
    return np.concatenate(pieces), sources


def read_series(cfg, data_root):
    spec = cfg['data']
    path = Path(data_root) / spec['path']
    if not path.exists():
        raise FileNotFoundError(f'Dataset not found: {path}. See README.md or use --data-root.')
    fmt = spec['format']
    if fmt == 'csv':
        frame = pd.read_csv(path)
        date = spec.get('date_column')
        if date and date in frame:
            times = pd.to_datetime(frame[date], errors='raise')
            if not times.is_monotonic_increasing or times.duplicated().any():
                if spec.get('timestamp_policy','strict')!='row_order':
                    raise ValueError(f'{path}: timestamps must be unique and increasing, or require an explicit row_order policy.')
                warnings.warn(f'{path.name}: non-monotonic/duplicate clock labels; preserving original observation row order as configured.',UserWarning)
        cols = [str(c) for c in frame.columns if c != date]
        last = spec.get('target_last')
        if last and last in cols:
            cols = [c for c in cols if c != last] + [last]
        values = frame[cols].to_numpy(dtype=np.float64)
        sources = [path]
    elif fmt == 'matrix':
        values = np.loadtxt(path, delimiter=',', dtype=np.float64, ndmin=2)
        cols, sources = [f'channel_{c}' for c in range(values.shape[1])], [path]
    elif fmt == 'windowed_npz':
        values, sources = reconstruct_windowed_npz(path)
        cols = [f'sensor_{c}' for c in range(values.shape[1])]
    else:
        raise ValueError(f'Unsupported format: {fmt}')
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Expected a finite [time,channel] matrix; missing values are not silently imputed.')
    return values, cols, sources


class SeriesData:
    def __init__(self, cfg, data_root=None, device='cpu', values=None):
        self.cfg, self.device = cfg, torch.device(device)
        self.L, self.H = int(cfg['model']['seq_len']), int(cfg['model']['pred_len'])
        H = self.H
        if values is None:
            raw, self.columns, sources = read_series(cfg, data_root)
        else:
            raw = np.asarray(values, dtype=np.float64)
            self.columns = [f'channel_{c}' for c in range(raw.shape[1])]
            sources = []
        split = cfg['data']['split']
        if split['type'] == 'fixed':
            tr, va, te = map(int, split['boundaries'])
        elif split['type'] == 'ratios':
            tr, te = int(len(raw)*split['train']), len(raw)
            va = te - int(te*split['test'])
        else:
            raise ValueError('Use explicit fixed boundaries or train/test ratios.')
        if not (self.L+H <= tr < va < te <= len(raw)) or min(va-tr, te-va) < H:
            raise ValueError(f'Insufficient data/invalid split: L={self.L}, H={H}, borders={(tr,va,te)}, rows={len(raw)}')
        self.borders, self.C = (tr, va, te), raw.shape[1]
        self.scaler = StandardScaler().fit(raw[:tr])
        z = self.scaler.transform(raw[:te]).astype(np.float32)
        self.z = torch.as_tensor(z, device=self.device).contiguous()
        self.zp = torch.cat([self.z[:1].expand(PAD, -1), self.z], 0)
        self.origins = dict(train=np.arange(self.L, tr-H+1), val=np.arange(tr, va-H+1), test=np.arange(va, te-H+1))
        m = cfg['memory']
        h = torch.arange(H, device=self.device)
        short = m['short_period'] * torch.ceil((h+1).float()/m['short_period']).long()
        lags = [short+k*m['short_period'] for k in range(7)]
        # For a period shorter than H (e.g. ExchangeRate's 20 observations),
        # choose the latest fully historical phase separately at each horizon.
        # This is exactly the legacy constant lag when long_period >= H.
        long = m['long_period'] * torch.ceil((h+1).float()/m['long_period']).long()
        lags += [long+k*m['long_period'] for k in range(4)]
        self.phase_lags = torch.stack(lags)
        if not (self.phase_lags >= h[None]+1).all():
            raise ValueError('Phase memory reads beyond the observed past.')
        self.provenance = dict(rows=len(raw),channels=self.C,columns=self.columns,borders=[tr,va,te],
              seq_len=self.L,pred_len=self.H,
              scaler_fit_half_open=[0,tr],scaler_mean=self.scaler.mean_.tolist(),scaler_scale=self.scaler.scale_.tolist(),
              origins={k:dict(first=int(v[0]),last=int(v[-1]),count=len(v)) for k,v in self.origins.items()},
              source_sha256={p.name:sha256(p) for p in sources},format=cfg['data']['format'],
              timestamp_policy=cfg['data'].get('timestamp_policy','strict'))

    def window(self, origins, length=None):
        t = torch.as_tensor(origins, device=self.device, dtype=torch.long).reshape(-1)
        length = self.L if length is None else length
        return self.zp[t[:,None]-length+torch.arange(length,device=self.device)+PAD]

    def target(self, origins):
        t = torch.as_tensor(origins, device=self.device, dtype=torch.long).reshape(-1)
        return self.z[t[:,None]+torch.arange(self.H,device=self.device)]

    def memory(self, origins):
        H = self.H
        t = torch.as_tensor(origins, device=self.device, dtype=torch.long).reshape(-1)
        B,C,dev = len(t),self.C,self.device
        tt = t[:,None,None,None]
        lags = self.phase_lags[None,None].expand(B,C,-1,-1)
        im = tt+torch.arange(H,device=dev)-lags
        ia = tt-1-lags
        ip = (tt-1-lags[...,:1])-torch.arange(KEY-1,-1,-1,device=dev)
        valid = (ip[...,0]>=0)&(ia.amin(-1)>=0)
        ci = torch.arange(C,device=dev)[None,:,None,None]
        def gather(indices):
            return self.zp[(indices+PAD).clamp(0,len(self.zp)-1),ci]
        return gather(im),gather(ia),gather(ip),valid
