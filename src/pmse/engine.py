"""Offline training and the unchanged full-H delayed-feedback FT deployment rule."""
import gc
import importlib.metadata
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from .calibration import MatureCalibrator, published_features
from .model import PMSE,BackboneOnly
from .statistics import compare
from .utils import fingerprint, load_json, run_lock, save_json, seed_all, sha256


def build_model(cfg, channels, device):
    fields=dict(cfg['model'],enc_in=channels,c_out=channels,backbone=cfg.get('backbone','PatchTST'))
    args=SimpleNamespace(**fields)
    model=BackboneOnly(args) if cfg.get('model_kind','pmse')=='backbone' else PMSE('P',args,cfg['memory']['width'])
    return model.to(device)


def forward(model,data,origins):
    memory=data.memory(origins) if model.needs_memory else None
    return model(data.window(origins),memory,x_mark=data.marks(origins))


@torch.no_grad()
def validation_mse(model,data,micro):
    model.eval()
    total,count=0.,0
    origins=data.origins['val']
    for s in range(0,len(origins),micro):
        t=origins[s:s+micro]
        se=(forward(model,data,t)-data.target(t)).square()
        total+=se.double().sum().item();count+=se.numel()
    return total/count


def train(cfg,data,seed,folder,device):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    summary_file=folder/'summary.json'
    if summary_file.exists():
        return load_json(summary_file)
    settings=cfg['training']
    batch,micro=settings['batch_size'],settings['micro_batch']
    results=[]
    new_lrs=settings['new_lrs'] if cfg.get('model_kind','pmse')=='pmse' else [0.]
    for ci,new_lr in enumerate(new_lrs):
        sub=folder/f'cfg{ci}';sub.mkdir(exist_ok=True)
        if (sub/'done.json').exists():
            results.append(load_json(sub/'done.json'));continue
        if (sub/'best.pt').exists() or (sub/'history.json').exists():
            raise RuntimeError(f'Partial training at {sub}; use a new --run-name after inspecting it. No silent restart.')
        seed_all(seed)
        model=build_model(cfg,data.C,device)
        groups=[dict(params=model.backbone_parameters(),lr=settings['backbone_lr'],base=settings['backbone_lr'])]
        if model.new_parameters():groups.append(dict(params=model.new_parameters(),lr=new_lr,base=new_lr))
        opt=torch.optim.Adam(groups)
        rng=np.random.default_rng(seed)
        best,best_ep,wait=float('inf'),0,0
        history=[];start=time.perf_counter()
        print(f'[{cfg["dataset"]} seed{seed}] cfg{ci} offline new_lr={new_lr:g}, micro_batch={micro}',flush=True)
        for ep in range(1,settings['max_epochs']+1):
            model.train()
            order=rng.permutation(data.origins['train'])
            train_sum,train_n=0.,0
            for s in range(0,len(order),batch):
                origins=order[s:s+batch]
                opt.zero_grad(set_to_none=True)
                # Splitting uses the legacy array_split partition, including the last
                # short optimizer batch. BN stats are deliberately microbatch stats.
                parts=min(int(np.ceil(batch/micro)),len(origins))
                for part in np.array_split(origins,parts):
                    loss=(forward(model,data,part)-data.target(part)).square().mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite offline loss.')
                    (loss*len(part)/len(origins)).backward()
                    train_sum+=loss.item()*len(part);train_n+=len(part)
                if ep==1 and s==0:
                    grads=[p.grad for p in model.parameters() if p.grad is not None]
                    if not grads or not all(torch.isfinite(g).all() for g in grads):
                        raise FloatingPointError('Missing/non-finite training gradients.')
                opt.step()
            val=validation_mse(model,data,micro)
            row=dict(epoch=ep,train_mse=train_sum/train_n,val_mse=val,lrs=[g['lr'] for g in opt.param_groups])
            history.append(row);save_json(sub/'history.json',history)
            print(f'  epoch {ep:03d}: train={row["train_mse"]:.8f} val={val:.8f}',flush=True)
            if val<best:
                best,best_ep,wait=val,ep,0
                temp=sub/'best.tmp';torch.save(model.state_dict(),temp);temp.replace(sub/'best.pt')
            else:
                wait+=1
                if wait>=settings['patience']:break
            for group in opt.param_groups:
                group['lr']=group['base'] if ep<3 else group['base']*.9**(ep-3)
        result=dict(config_index=ci,new_lr=new_lr,val_mse=best,best_epoch=best_ep,epochs=len(history),
                    seconds=time.perf_counter()-start,micro_batch=micro,optimizer_batch=batch,
                    checkpoint_sha256=sha256(sub/'best.pt'))
        save_json(sub/'done.json',result);results.append(result)
        del model,opt;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    selected=min(range(len(results)),key=lambda i:results[i]['val_mse'])
    shutil.copy2(folder/f'cfg{selected}'/'best.pt',folder/'selected.pt')
    summary=dict(seed=seed,selected=selected,selected_by='validation_mse_only',configurations=results,
                 checkpoint='selected.pt',seconds=sum(x['seconds'] for x in results))
    save_json(summary_file,summary)
    return summary


def _metric_groups(se_h,se_c,origins,columns,cfg):
    # [methods,N,H/C]; PFT,PFTC for PMSE or BFT for native baselines.
    window=np.asarray(se_h).mean(-1)
    n=window.shape[1]
    top=np.argsort(window[0],kind='stable')[-max(1,int(np.ceil(.1*n))):]
    high=np.zeros(n,dtype=bool);high[top]=True
    groups={}
    for label,mask in [('high_error_10pct',high),('ordinary_90pct',~high)]:
        groups[label]=dict(count=int(mask.sum()),mse=window[:,mask].mean(1).tolist() if mask.any() else None)
    blocks=[]
    for i in range(0,n,256):
        values=window[:,i:i+256].mean(1)
        blocks.append(dict(first_origin=int(origins[i]),last_origin=int(origins[min(i+255,n-1)]),
                           count=min(256,n-i),mse=values.tolist(),C_wins=bool(values[1]<values[0]) if len(values)>1 else None))
    ans=dict(window_groups=groups,blocks256=blocks,block_wins=sum(x['C_wins'] for x in blocks) if len(window)>1 else None)
    if cfg['dataset']=='ETTh2' and 'LULL' in columns:
        keep=[i for i,c in enumerate(columns) if c!='LULL']
        ans['without_LULL']=dict(mse=np.asarray(se_c)[:,:,keep].mean((1,2)).tolist())
    return ans


def online(cfg,data,seed,folder,checkpoint,device,save_records=False):
    H = int(cfg['model']['pred_len'])
    if H != data.H:
        raise ValueError('Model and data prediction horizons must agree.')
    folder=Path(folder)
    if (folder/'summary.json').exists():
        if save_records and not (folder/'records'/'pred.npy').exists():
            raise RuntimeError('This completed run did not save raw records. Choose a new --run-name to collect them.')
        return load_json(folder/'summary.json')
    folder.mkdir(parents=True,exist_ok=True)
    if any(folder.iterdir()):
        raise RuntimeError(f'Partial online output at {folder}; inspect it and use a new --run-name.')
    model=build_model(cfg,data.C,device)
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
    model.eval()
    opt=torch.optim.Adam(model.parameters(),lr=cfg['online']['lr'])
    seed_all(seed)
    stream=np.concatenate([data.origins['val'],data.origins['test']])
    nv,nt=len(data.origins['val']),len(data.origins['test'])
    use_calibration=cfg.get('model_kind','pmse')=='pmse'
    methods=['PFT','PFTC'] if use_calibration else ['BFT']
    nm=len(methods)
    cal=MatureCalibrator(data.C,H,cfg['calibration']['n0'],cfg['calibration']['eps']) if use_calibration else None
    se_h=np.lib.format.open_memmap(folder/'squared_error_horizon.npy',mode='w+',dtype='float64',shape=(nm,nt,H))
    se_c=np.lib.format.open_memmap(folder/'squared_error_channel.npy',mode='w+',dtype='float64',shape=(nm,nt,data.C))
    totals=np.zeros(nm);abs_totals=np.zeros(nm);count=0
    supports=np.full(len(stream),-1,dtype=np.int64);counts=np.zeros(len(stream),dtype=np.int64)
    records={}
    if save_records:
        rec=folder/'records';rec.mkdir()
        shapes={k:(len(stream),data.C,H) for k in ('y','pD','pW')} if use_calibration else {}
        if use_calibration:shapes.update(mu=(len(stream),data.C),sd=(len(stream),data.C))
        shapes.update(pred=(len(stream),H,data.C),truth=(len(stream),H,data.C))
        records={k:np.lib.format.open_memmap(rec/f'{k}.npy',mode='w+',dtype='float32',shape=v) for k,v in shapes.items()}
    start=time.perf_counter();checks=dict(max_reconstruction_error=0.,max_alpha_sum_error=0.)
    def mature_target(j):
        return data.target([j])[0].detach().cpu().numpy()
    for i,t in enumerate(stream):
        t=int(t);j=t-H
        # Full H labels are available at t; x_j and memory_j remain historical.
        assert j+H<=t
        if cal is not None:cal.mature(t,mature_target)
        opt.zero_grad(set_to_none=True)
        loss=(forward(model,data,[j])-data.target([j])).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite online FT loss.')
        loss.backward();opt.step()
        with torch.no_grad():
            if use_calibration:
                p,f,c=published_features(model,data.window([t]),data.memory([t]),data.marks([t]))
            else:p,f,c=forward(model,data,[t]),{},{}
        pred=p[0].float().cpu().numpy()
        feature={k:v[0].detach().float().cpu().numpy() for k,v in f.items()}
        predictions=(pred,)
        if use_calibration:
            corrected=cal.predict_and_publish(t,feature)
            predictions=(pred,corrected)
            supports[i],counts[i]=cal.latest,cal.count
            checks['max_reconstruction_error']=max(checks['max_reconstruction_error'],c['reconstruction'])
            checks['max_alpha_sum_error']=max(checks['max_alpha_sum_error'],c['alpha_sum'])
        # Labels for scoring are accessed after publication and never passed to an
        # optimizer until the corresponding complete H-step forecast has matured.
        truth=mature_target(t)
        if records:
            for k,v in feature.items():records[k][i]=v
            records['pred'][i]=pred;records['truth'][i]=truth
        if i>=nv:
            pos=i-nv
            for method,value in enumerate(predictions):
                error=value.astype(np.float64)-truth.astype(np.float64)
                square=error*error
                se_h[method,pos]=square.mean(1);se_c[method,pos]=square.mean(0)
                totals[method]+=square.sum();abs_totals[method]+=np.abs(error).sum()
            count+=truth.size
        if i%500==0 or i==len(stream)-1:
            print(f'[{cfg["dataset"]} seed{seed}] online {i+1}/{len(stream)}, C_support={cal.count if cal is not None else "disabled"}',flush=True)
    se_h.flush();se_c.flush()
    for value in records.values():value.flush()
    if checks['max_reconstruction_error']>=1e-4 or checks['max_alpha_sum_error']>=1e-4:
        raise AssertionError(f'Feature reconstruction failed: {checks}')
    keep=supports>=0
    assert np.all(supports[keep]+H<=stream[keep])
    np.savez(folder/'support_schedule.npz',origins=stream,n_validation=nv,pred_len=H,FT_support=stream-H,
             C_latest_support=supports,C_count=counts)
    stats=cfg['statistics']
    comparison=compare(se_h[1].mean(1),se_h[0].mean(1),stats['block_length'],stats['bootstrap_draws'],stats['seed']) if use_calibration else None
    result=dict(dataset=cfg['dataset'],seed=seed,pred_len=H,backbone=cfg.get('backbone','PatchTST'),model_kind=cfg.get('model_kind','pmse'),scope='full100',n_excluded=0,n_test=nt,columns=data.columns,
                metric_scale='train_standardized',methods=methods,
                mse=dict(zip(methods,(totals/count).tolist())),
                mae=dict(zip(methods,(abs_totals/count).tolist())),
                segment_mse=np.asarray(se_h).reshape(nm,nt,4,H//4).mean((1,3)).tolist(),
                segment_ranges=[[g*H//4+1,(g+1)*H//4] for g in range(4)],
                channel_mse=np.asarray(se_c).mean(1).tolist(),comparison=comparison,
                seconds=time.perf_counter()-start,raw_records_saved=save_records,
                checks=dict(**checks,full_H_maturity=True,FT_update_before_prediction=True,
                            C_enabled=use_calibration,C_uses_published_features=use_calibration,C_does_not_change_FT=True,
                            train_only_scaler=True,validation_warmup=True,BN_eval_online=True),
                **_metric_groups(se_h,se_c,data.origins['test'],data.columns,cfg))
    save_json(folder/'summary.json',result)
    del model,opt,se_h,se_c,records;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return result


def run_seed(cfg,data,seed,folder,device,stage='all',save_records=False):
    folder=Path(folder)
    package=Path(__file__).parent
    implementation={p.relative_to(package).as_posix():sha256(p) for p in package.rglob('*.py')}
    spec=dict(config=cfg,seed=seed,data=data.provenance,implementation=implementation)
    identity=fingerprint(spec)
    with run_lock(folder):
        config_path=folder/'run.json'
        if config_path.exists():
            if load_json(config_path)['fingerprint']!=identity:
                raise RuntimeError('Config/data/code drift. Choose a new --run-name; existing outputs are preserved.')
        else:
            runtime=dict(python=sys.version.split()[0],torch=torch.__version__,device=str(device),
                         numpy=np.__version__,pandas=importlib.metadata.version('pandas'),
                         scikit_learn=importlib.metadata.version('scikit-learn'))
            save_json(config_path,dict(fingerprint=identity,runtime=runtime,**spec))
        checkpoint=folder/'offline/selected.pt'
        start=time.perf_counter()
        if stage in ('all','offline'):
            train(cfg,data,seed,folder/'offline',device)
        if stage in ('all','online'):
            if not checkpoint.exists():raise FileNotFoundError(f'Train first: {checkpoint}')
            result=online(cfg,data,seed,folder/'online',checkpoint,device,save_records)
            save_json(folder/'complete.json',dict(fingerprint=identity,dataset=cfg['dataset'],seed=seed,
                      pred_len=data.H,backbone=cfg['backbone'],model_kind=cfg['model_kind'],
                      scope='full100',mse=result['mse'],mae=result['mae'],invocation_seconds=time.perf_counter()-start))
            return result
    return None
