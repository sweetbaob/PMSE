"""PROCEED TCN: TS2Vec encoder and original multivariate forecast head."""
import torch
from torch import nn
from .ts2vec_encoder import TS2VecEncoderWrapper, TSEncoder

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        output_dims = getattr(args, "tcn_output_dims", 320)
        encoder = TSEncoder(input_dims=args.enc_in + 7,
                            output_dims=output_dims,  # standard ts2vec backbone value
                            hidden_dims=getattr(args, "tcn_hidden_dims", 64),  # standard ts2vec backbone value
                            depth=getattr(args, "tcn_depth", 10))
        self.encoder = TS2VecEncoderWrapper(encoder, mask='all_true')
        self.pred_len = args.pred_len
        self.dim = args.c_out * args.pred_len

        # self.regressor = nn.Sequential(nn.Linear(320, 320), nn.ReLU(), nn.Linear(output_dims, self.dim)).to(self.device)
        self.regressor = nn.Linear(output_dims, self.dim)

    def forward(self, x, x_mark=None):
        if x_mark is None:
            x_mark = torch.zeros(*x.shape[:2], 7, device=x.device)
        x = torch.cat([x, x_mark], dim=-1)
        rep = self.encoder(x)
        y = self.regressor(rep)
        y = y.reshape(len(y), self.pred_len, -1)
        return y
