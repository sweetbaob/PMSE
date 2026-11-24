"""C V2.1: streaming sufficient statistics in train-standardized data units."""
from collections import deque
import numpy as np
import torch

SHORT_INDICES = list(range(3,10)) + list(range(14,21))
LONG_INDICES = list(range(10,14)) + list(range(21,25))


def published_features(model, x, memory, x_mark=None):
    """Export features without changing the PMSE prediction or gradient graph."""
    # The full model decode adds no trainable parameters; the subclass only exports
    # candidates that are already part of the original forward computation.
    out, aux = model(x, memory, return_aux=True, x_mark=x_mark)
    rv = model.normalization()
    mu, sd = rv.mean[:,0,:], rv.stdev[:,0,:]
    y, alpha, candidates = aux['y'], aux['alpha'], aux['V'].transpose(2,3)
    def proposal(indices):
        w = alpha[...,indices]
        mass = w.sum(-1)
        p = (w*candidates[...,indices]).sum(-1)/mass.clamp_min(1e-12)
        return torch.where(mass>1e-6,p-y,torch.zeros_like(y))
    feature = dict(y=y,pD=proposal(SHORT_INDICES),pW=proposal(LONG_INDICES),mu=mu,sd=sd)
    recon = y.permute(0,2,1)*sd[:,None]+mu[:,None]
    checks = dict(reconstruction=float((recon-out).abs().max().detach()),alpha_sum=float((alpha.sum(-1)-1).abs().max().detach()))
    return out,feature,checks


class MatureCalibrator:
    def __init__(self, channels, horizon=48, n0=168, eps=1e-8):
        self.H,self.n0,self.eps=horizon,n0,eps
        self.stats=np.zeros((channels,horizon,5),dtype=np.float64)
        self.pending=deque()
        self.count=0
        self.latest=-1
        self.last_published=-1

    @staticmethod
    def components(feature):
        mu,sd=feature['mu'].astype(np.float64),feature['sd'].astype(np.float64)
        y=feature['y'].astype(np.float64)
        p=np.stack([feature['pD'],feature['pW']],-1).astype(np.float64)*sd[...,None,None]
        return mu,sd,y,p

    def mature(self, origin, target_fn):
        while self.pending and self.pending[0][0]+self.H <= origin:
            j,f=self.pending.popleft()
            mu,sd,y,p=self.components(f)
            truth=np.asarray(target_fn(j),dtype=np.float64).T
            e=((truth-mu[...,None])/sd[...,None]-y)*sd[...,None]
            u,v=p[...,0],p[...,1]
            self.stats += np.stack([u*u,u*v,v*v,u*e,v*e],-1)
            self.count+=1
            self.latest=j
        if self.latest>=0 and self.latest+self.H>origin:
            raise AssertionError('Unmatured feedback entered C.')

    def predict_and_publish(self, origin, feature):
        if origin<=self.last_published:
            raise ValueError('Published origins must be strictly increasing.')
        mu,sd,y,p=self.components(feature)
        k=np.zeros((*y.shape,2),dtype=np.float64)
        if self.count:
            a=self.stats
            a11=a[...,0]*(1+self.n0/self.count)+self.eps
            a22=a[...,2]*(1+self.n0/self.count)+self.eps
            a12=a[...,1]
            det=a11*a22-a12*a12
            if not np.isfinite(det).all() or np.any(det<=0):
                raise FloatingPointError('Non-positive C normal-equation determinant; no silent clipping.')
            k[...,0]=(a22*a[...,3]-a12*a[...,4])/det
            k[...,1]=(a11*a[...,4]-a12*a[...,3])/det
        pred=(y*sd[...,None]+mu[...,None]+(k*p).sum(-1)).T
        self.pending.append((int(origin),{key:np.array(value,copy=True) for key,value in feature.items()}))
        self.last_published=int(origin)
        return pred
