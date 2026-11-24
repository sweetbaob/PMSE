"""Phase-memory candidate fusion, retaining the complete PatchTST head."""
import math
import torch
from torch import nn
from .backbones.patchtst import Model as PatchTST
from .backbones.revin import RevIN
from .backbones.registry import make_native,feature_dim
KEY, NDAY, NWEEK = 16, 7, 4
S_SLOTS = NDAY + NWEEK
VARIANTS = ('P',)

class PMSE(nn.Module):
    def __init__(self, variant, cfg, width):
        super().__init__()
        assert variant in VARIANTS
        self.variant = variant
        self.H = H = int(cfg.pred_len)
        self.backbone_name = getattr(cfg,'backbone','PatchTST')
        self.needs_memory = True
        if self.backbone_name=='PatchTST':
            self.pt = PatchTST(cfg)
            net = self.pt.model
            assert net.revin and not net.revin_layer.affine and not net.revin_layer.subtract_last
            assert not self.pt.decomposition
        else:
            self.core = make_native(self.backbone_name,cfg)
            self.revin = RevIN(cfg.enc_in,affine=False,subtract_last=False)
            if self.backbone_name=='iTransformer':self.core.use_norm=False
        S = S_SLOTS
        self.use_state = True
        self.use_slots = True
        self.value_path = True
        nc = 1 + (2 if self.use_state else 0) + (2 * S if self.use_slots else 0)
        d = width
        self.heads = 4
        self.tok = nn.Sequential(nn.Linear(KEY + H + 1, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))
        self.typ = nn.Parameter(torch.randn(nc, d) * 0.02)
        self.qin = nn.Sequential(nn.Linear(2 * feature_dim(self.backbone_name,cfg), d), nn.GELU(), nn.Linear(d, d))
        self.hemb = nn.Parameter(torch.randn(H, d) * 0.02)
        self.qln = nn.LayerNorm(d)
        self.wq, self.wk, self.wv = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.fout = nn.Linear(d, 1)
        nn.init.zeros_(self.fout.weight)
        nn.init.zeros_(self.fout.bias)
        self.gate = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.constant_(self.gate[-1].bias, -2.0)
        self.drop = nn.Dropout(cfg.dropout)

    def new_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith(('pt.','core.','revin.'))]

    def backbone_parameters(self):
        return list((self.pt if self.backbone_name=='PatchTST' else self.core).parameters())

    def normalization(self):
        return self.pt.model.revin_layer if self.backbone_name=='PatchTST' else self.revin

    def _forecast_features(self,x,x_mark=None):
        """Forecast and native features in this query's outer RevIN coordinates."""
        rv=self.normalization()
        xn=rv(x,'norm')
        if self.backbone_name=='PatchTST':
            net=self.pt.model
            z=xn.permute(0,2,1)
            if net.padding_patch=='end':z=net.padding_patch_layer(z)
            z=z.unfold(-1,net.patch_len,net.stride).permute(0,1,3,2)
            Z=net.backbone(z)
            yb=net.head(Z)
        elif self.backbone_name=='TCN':
            marks=x_mark if x_mark is not None else xn.new_zeros(*xn.shape[:2],7)
            rep=self.core.encoder(torch.cat([xn,marks],dim=-1))
            yb=self.core.regressor(rep).reshape(len(xn),self.H,-1).transpose(1,2)
            Z=rep[:,None,:,None].expand(-1,xn.shape[-1],-1,-1)
        elif self.backbone_name=='iTransformer':
            y,rep=self.core(xn,x_mark,return_emb=True)
            yb=y.transpose(1,2)
            Z=rep[:,:xn.shape[-1],:,None]
        else:
            yb=self.core(xn).transpose(1,2)
            # Linear models have no latent encoder. Use the full observed input
            # directly, with no learned compression or additional prediction head.
            Z=xn.transpose(1,2).unsqueeze(-1)
        return xn,Z,yb

    def forward(self, x, mem=None, return_aux=False, x_mark=None):
        if mem is None:
            raise ValueError('PMSE requires causal phase memory from SeriesData.memory().')
        xn,Z,yb = self._forecast_features(x,x_mark)
        rv = self.normalization()
        mu = rv.mean.transpose(1, 2)
        sd = rv.stdev.transpose(1, 2)
        xl = xn.permute(0, 2, 1)                                 # [B, C, L]
        last = xl[..., -1:]
        m, a, p, valid = mem
        m = (m - mu[..., None]) / sd[..., None]
        a = (a - mu[..., None]) / sd[..., None]
        p = (p - mu[..., None]) / sd[..., None]
        vm = valid[..., None]
        m = torch.where(vm, m, last[..., None])
        a = torch.where(vm, a, last[..., None])
        p = torch.where(vm, p, xl[..., None, -KEY:])
        y, aux = self.decode(xl, Z, yb, m, a, p, valid)
        out = rv(y.permute(0, 2, 1), 'denorm')          # [B, H, C]
        return (out, aux) if return_aux else out

    def decode(self, xl, Z, yb, m, a, p, valid):
        H = self.H
        B, C, _ = yb.shape
        dev = yb.device
        last = xl[..., -1:]
        x16 = xl[..., -KEY:]
        zero16, zero1 = torch.zeros_like(x16), torch.zeros_like(last)
        ones = torch.ones(B, C, 1, dtype=torch.bool, device=dev)
        V = [yb[:, :, None]]
        F = [torch.cat([zero16, yb - last, zero1], -1)[:, :, None]]
        M = [ones]
        if self.use_state:
            mean24 = xl[..., -24:].mean(-1, keepdim=True)
            V += [last.expand(-1, -1, H)[:, :, None], mean24.expand(-1, -1, H)[:, :, None]]
            F += [torch.zeros(B, C, 1, KEY + H + 1, device=dev),
                  torch.cat([zero16, (mean24 - last).expand(-1, -1, H), mean24 - last], -1)[:, :, None]]
            M += [ones, ones]
        if self.use_slots:
            pdiff = p - x16[:, :, None]
            lev = a[..., :1] - last[..., None]
            V += [m, last[..., None] + m - a]
            F += [torch.cat([pdiff, m - last[..., None], lev], -1), torch.cat([pdiff, m - a, lev], -1)]
            M += [valid, valid]
        V, F, M = torch.cat(V, 2), torch.cat(F, 2), torch.cat(M, 2)
        N, nc = B * C, V.shape[2]
        V, F, M = V.reshape(N, nc, H), F.reshape(N, nc, -1).clamp(-10, 10), M.reshape(N, nc)
        U = self.drop(self.tok(F) + self.typ)                        # [N, nc, d]
        Zf = Z.reshape(N, Z.shape[2], Z.shape[3])
        Q = self.qln(self.qin(torch.cat([Zf.mean(-1), Zf[..., -1]], -1))[:, None] + self.hemb)   # [N, H, d]
        nh, d = self.heads, Q.shape[-1]
        dk = d // nh
        q_ = self.wq(Q).view(N, H, nh, dk).transpose(1, 2)
        k_ = self.wk(U).view(N, nc, nh, dk).transpose(1, 2)
        v_ = self.wv(U).view(N, nc, nh, dk).transpose(1, 2)
        lg = (q_ @ k_.transpose(-1, -2)) / math.sqrt(dk)             # [N, nh, H, nc]
        al = lg.masked_fill(~M[:, None, None, :], -1e4).softmax(-1)
        ab = al.mean(1)                                              # [N, H, nc]
        yf = yb.reshape(N, H)
        r = torch.einsum('nhc,nch->nh', ab, V - yf[:, None]) if self.value_path else torch.zeros_like(yf)
        f = self.fout((al @ v_).transpose(1, 2).reshape(N, H, d)).squeeze(-1)
        g = torch.sigmoid(self.gate(Q).squeeze(-1))
        y = yf + g * (r + f)
        # Export existing candidates for C; no extra parameters or prediction path.
        aux = dict(alpha=ab.view(B, C, H, nc), gate=g.view(B, C, H),
                   V=V.view(B, C, nc, H), y=y.view(B, C, H))
        return y.view(B, C, H), aux


class BackboneOnly(nn.Module):
    """Native forecast + delayed FT, with no A/B, C or added nonlinear head."""
    needs_memory=False

    def __init__(self,cfg):
        super().__init__()
        self.backbone_name=cfg.backbone
        self.core=make_native(self.backbone_name,cfg)

    def forward(self,x,mem=None):
        if self.backbone_name=='iTransformer':return self.core(x,None)
        return self.core(x)

    def backbone_parameters(self):return list(self.core.parameters())

    def new_parameters(self):return []

