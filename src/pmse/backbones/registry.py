"""Copied native forecasting models; imports remain lazy for standalone users."""
SUPPORTED_BACKBONES = ('PatchTST','TCN','iTransformer','DLinear')


def make_native(name,cfg):
    if name=='PatchTST':
        from .patchtst import Model
    elif name=='TCN':
        from .tcn import Model
    elif name=='iTransformer':
        from .itransformer import Model
    elif name=='DLinear':
        from .dlinear import Model
    else:
        raise ValueError(f'Unknown backbone {name!r}; choose {SUPPORTED_BACKBONES}.')
    return Model(cfg)


def feature_dim(name,cfg):
    if name=='TCN':return getattr(cfg,'tcn_output_dims',320)
    if name=='DLinear':return cfg.seq_len
    return cfg.d_model
