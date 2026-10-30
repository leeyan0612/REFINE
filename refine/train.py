"""Train a frozen-feature REFINE head from disjoint prepared CORE/META streams."""
import argparse
import json
from pathlib import Path
import torch
from .model import REFINE
from .training import make_controller,make_optimizer,training_step

def move(x,device):
    if torch.is_tensor(x):return x.to(device)
    if isinstance(x,dict):return {k:move(v,device) for k,v in x.items()}
    if isinstance(x,list):return [move(v,device) for v in x]
    if isinstance(x,tuple):return tuple(move(v,device) for v in x)
    return x

def main():
    p=argparse.ArgumentParser()
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--bundle',help='JSON with model dimensions and prepared batch paths')
    source.add_argument('--annotations',help='Released annotations/manifest.json')
    p.add_argument('--feature-root',help='Per-image frozen feature NPZ files, named by public_id')
    p.add_argument('--text-bank',help='Frozen EyeCLIP NPZ with train [64,3,512]')
    p.add_argument('--local-dim',type=int,default=768)
    p.add_argument('--global-dim',type=int,default=512)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--seed',type=int,default=1901)
    p.add_argument('--static',action='store_true')
    a=p.parse_args()
    if torch.device(a.device).type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
    bank=None
    if a.annotations:
        from .data import AnnotationBank,Plans
        if not a.feature_root or not a.text_bank:p.error('--annotations requires --feature-root and --text-bank')
        bank=AnnotationBank(a.annotations,a.feature_root,a.text_bank,a.device)
        core_groups={bank.records[i]['content_group']for i in bank.core}
        meta_groups={bank.records[i]['content_group']for i in bank.meta}
        assert core_groups.isdisjoint(meta_groups)
        c=dict(local_dim=a.local_dim,global_dim=a.global_dim,core_content_groups=sorted(core_groups),
               meta_content_groups=sorted(meta_groups))
        core_stream=Plans(a.seed,bank.core,bank.allowed)
        meta_stream=Plans(a.seed+25001,bank.meta,bank.allowed)
    else:
        root=Path(a.bundle).resolve().parent;c=json.loads(Path(a.bundle).read_text())
        assert c['core_content_groups'] and c['meta_content_groups']
        assert set(c['core_content_groups']).isdisjoint(c['meta_content_groups'])
    torch.manual_seed(a.seed);model=REFINE(c['local_dim'],c['global_dim']).to(a.device)
    optimizer=make_optimizer(model);controller=None if a.static else make_controller(a.device)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    for step in range(1,a.steps+1):
        core=(bank.prepare(core_stream.next(),'CORE')if bank is not None else
              move(torch.load(root/c['core_batches'][(step-1)%len(c['core_batches'])],
                              map_location='cpu',weights_only=True),a.device))
        meta=None
        if controller is not None and step%10==0:
            meta=(bank.prepare(meta_stream.next(),'META')if bank is not None else
                  move(torch.load(root/c['meta_batches'][(step//10-1)%len(c['meta_batches'])],
                                  map_location='cpu',weights_only=True),a.device))
        info=training_step(model,core,optimizer,step=step,controller=controller,meta=meta)
        if step%100==0:print(json.dumps({'step':step,'loss':info['loss'],'axis_weights':info['axis_weights']}))
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'model.pt')
    (out/'config.json').write_text(json.dumps(dict(c,seed=a.seed,steps=a.steps,static=a.static),indent=2))

if __name__=='__main__':main()
