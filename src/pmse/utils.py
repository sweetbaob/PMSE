"""Portable paths, reproducible seeds, atomic outputs and exclusive run locks."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import numpy as np
import torch
from .backbones.registry import SUPPORTED_BACKBONES

PROJECT = Path(__file__).resolve().parents[2]
SUPPORTED_HORIZONS = (12, 24, 48, 96)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def run_lock(folder):
    """Atomic directory creation works on Windows/Linux; never silently steal a lock."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / '.run.lock'
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(f'Run is locked: {lock}. Inspect owner.json and the process before removing a stale lock.') from None
    save_json(lock / 'owner.json', dict(pid=os.getpid(), host=socket.gethostname()))
    try:
        yield
    finally:
        (lock / 'owner.json').unlink()
        lock.rmdir()


def load_config(dataset, pred_len=None, backbone=None, model_kind=None):
    path = Path(dataset)
    if not path.is_file():
        path = PROJECT / 'configs' / f'{dataset}.json'
    if not path.is_file():
        raise ValueError(f'Unknown dataset/config: {dataset}')
    cfg = load_json(path)
    name = backbone or cfg.get('backbone','PatchTST')
    if name not in SUPPORTED_BACKBONES:
        raise ValueError(f'Unknown backbone {name!r}; choose {SUPPORTED_BACKBONES}.')
    previous_preset = cfg.get('backbone_preset',{}).get('name')
    if previous_preset and previous_preset != name:
        if 'backbone_base_model' not in cfg:
            raise ValueError('To switch backbones, use a dataset JSON or a config containing backbone_base_model.')
        # Do not carry iTransformer widths/dropout into a later PatchTST run.
        lengths = {k:cfg['model'][k] for k in ('seq_len','pred_len')}
        cfg['model'] = dict(cfg['backbone_base_model'],**lengths)
    cfg.setdefault('backbone_base_model',dict(cfg['model']))
    cfg['backbone'] = name
    cfg['model_kind'] = model_kind or cfg.get('model_kind','pmse')
    if cfg['model_kind'] not in ('pmse','backbone'):
        raise ValueError('model_kind must be pmse or backbone.')
    preset = load_json(PROJECT/'configs/backbones'/f'{name}.json')
    # A versioned preset is applied only when selecting a different backbone;
    # complete custom JSON configurations remain authoritative on later loads.
    if 'backbone_preset' not in cfg or cfg['backbone_preset'].get('name') != name:
        cfg['model'].update(preset.get('model',{}))
        cfg['model'].update(preset.get('dataset_model_overrides',{}).get(cfg['dataset'],{}))
    cfg['backbone_preset'] = preset
    if pred_len is not None:
        cfg['model']['pred_len'] = pred_len
    if cfg['model']['pred_len'] not in SUPPORTED_HORIZONS:
        raise ValueError(f'pred_len must be one of {SUPPORTED_HORIZONS}.')
    m = cfg['memory']
    if (m['short_slots'], m['long_slots'], m['key_length']) != (7, 4, 16):
        raise ValueError('The released architecture uses 7 short, 4 long slots and 16-point keys.')
    return cfg
