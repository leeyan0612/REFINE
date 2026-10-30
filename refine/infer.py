"""Inference with frozen image features and EyeCLIP sentence embeddings."""
import argparse
import contextlib
from pathlib import Path
import numpy as np
import torch
from .model import load_model

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',required=True)
    p.add_argument('--features',required=True,help='NPZ: local, global_features, text_vectors, support')
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu')
    a=p.parse_args();model=load_model(a.weights,a.device)
    with np.load(a.features,allow_pickle=False) as z:
        local=torch.as_tensor(z['local'],device=a.device)
        glob=torch.as_tensor(z['global_features'],device=a.device)
        text=torch.as_tensor(z['text_vectors'],device=a.device)
        support=torch.as_tensor(z['support'],device=a.device,dtype=torch.bool)
    precision=(torch.autocast('cuda',dtype=torch.bfloat16)
               if torch.device(a.device).type=='cuda'else contextlib.nullcontext())
    with torch.inference_mode(),precision:out=model(local,glob,text,support)
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,heatmaps=out['heatmap_probabilities'].cpu().numpy(),
        retrieval_scores=out['retrieval_scores'].cpu().numpy(),
        latent_points=out['anchors01'].float().cpu().numpy(),
        latent_scales=torch.stack((out['dd01'],out['radius01']),-1).float().cpu().numpy())

if __name__=='__main__':main()
