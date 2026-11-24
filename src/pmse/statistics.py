"""Paired fixed-length moving-block bootstrap and transparent subgroup summaries."""
import numpy as np


def compare(a,b,block_length,draws=2000,seed=0):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    if len(a)!=len(b) or len(a)==0:
        raise ValueError('Paired nonempty window losses required.')
    n=len(a)
    block_length=min(block_length,n)
    k=int(np.ceil(n/block_length))
    starts=np.random.default_rng(seed).integers(0,n,size=(draws,k))
    lengths=np.full(k,block_length);lengths[-1]=n-block_length*(k-1)
    def resample(x):
        cs=np.concatenate([[0.],np.cumsum(np.concatenate([x,x]))])
        return (cs[starts+lengths]-cs[starts]).sum(1)
    denom=resample(b)
    if b.sum()<=0 or np.any(denom<=0):
        return dict(relative_mse=None,ci95=None,reason='zero baseline error')
    boot=resample(a)/denom-1
    thirds=[float(a[s].sum()/b[s].sum()-1) if b[s].sum()>0 else None for s in np.array_split(np.arange(n),3)]
    return dict(relative_mse=float(a.sum()/b.sum()-1),ci95=np.percentile(boot,[2.5,97.5]).tolist(),
                thirds=thirds,block_length=block_length,draws=draws,seed=seed)
