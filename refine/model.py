"""REFINE frozen-feature query head and image-only latent geometry."""
import math
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F
N_FAMILIES=17
PROTOTYPE_SEED=2026102011
GEOMETRY_EPS=1e-4
D_MAX=2/math.sqrt(math.pi)
R_MAX=math.sqrt(2)
def setting(arm):
    if arm != 'TEXT28_BASE': raise ValueError('This release uses the 28x28 text-query head')
    return dict(feature_grid=28,heatmap=256,text_mode='shared_frozen_eyeclip')
def orthogonal_prototypes(seed=PROTOTYPE_SEED):
    generator=torch.Generator(device='cpu').manual_seed(seed)
    q,_=torch.linalg.qr(torch.randn(512,N_FAMILIES,generator=generator,dtype=torch.float64),mode='reduced')
    return q.T.contiguous().float()

def window_gather_indices(size=256,grid=28):
    pieces=[]
    for y in range(grid):
        a=max(0,math.ceil(y*size/grid-.5));b=min(size,math.ceil((y+1)*size/grid-.5))
        for x in range(grid):
            c=max(0,math.ceil(x*size/grid-.5));d=min(size,math.ceil((x+1)*size/grid-.5))
            pieces.append([j*size+i for j in range(a,b)for i in range(c,d)])
    indices=torch.zeros(grid*grid,max(map(len,pieces)),dtype=torch.long);inside=torch.zeros_like(indices,dtype=torch.bool)
    for i,piece in enumerate(pieces):indices[i,:len(piece)]=torch.tensor(piece);inside[i,:len(piece)]=True
    return indices,inside

def score_windows(probabilities,valid_support256=None):
    assert probabilities.ndim==4 and probabilities.shape[-2:]in((256,256),(512,512))
    common=F.avg_pool2d(probabilities,2,2)if probabilities.shape[-1]==512 else probabilities
    indices,inside=window_gather_indices();indices=indices.to(common.device);inside=inside.to(common.device)
    b,q=common.shape[:2];values=common.flatten(2)[:,:,indices.flatten()].reshape(b,q,784,-1)
    supported=inside[None].expand(b,-1,-1)
    if valid_support256 is not None:
        assert valid_support256.shape==(b,256,256)
        supported=supported&(valid_support256.flatten(1)[:,indices.flatten()].reshape(b,784,-1)>0)
    available=supported.sum(-1);top=values.masked_fill(~supported[:,None],-torch.inf).topk(4,dim=-1).values
    return torch.where(torch.isfinite(top),top,torch.zeros_like(top)).sum(-1)/available.clamp(min=1,max=4)[:,None]

def predicted_geometry(anchors01,dd01,radius01,side):
    assert anchors01.shape==(len(dd01),2,2)and dd01.shape==radius01.shape==(len(anchors01),)
    grid=(torch.arange(side,device=anchors01.device,dtype=anchors01.dtype)+.5)/side
    yy,xx=torch.meshgrid(grid,grid,indexing='ij');xy=torch.stack((xx,yy),-1).reshape(1,side*side,2)
    relative=xy-anchors01[:,1,None];D=dd01.clamp_min(GEOMETRY_EPS);R=radius01.clamp_min(GEOMETRY_EPS)
    axis=anchors01[:,0]-anchors01[:,1];length=axis.norm(dim=-1);axis_valid=length>GEOMETRY_EPS
    fallback=torch.tensor([1.,0.],device=axis.device,dtype=axis.dtype).expand_as(axis)
    disc=torch.where(axis_valid[:,None],axis/length.clamp_min(GEOMETRY_EPS)[:,None],fallback)
    perpendicular=torch.stack((disc[:,1],-disc[:,0]),-1)
    perpendicular=torch.where((perpendicular[:,1]<=0)[:,None],perpendicular,-perpendicular)
    distance=relative.norm(dim=-1);along=(relative*disc[:,None]).sum(-1);across=(relative*perpendicular[:,None]).sum(-1)
    phi=torch.stack((relative[:,:,0]/D[:,None],relative[:,:,1]/D[:,None],distance/D[:,None],along/D[:,None],across/D[:,None],
        along/distance.clamp_min(GEOMETRY_EPS),across/distance.clamp_min(GEOMETRY_EPS),
        dd01[:,None].expand_as(distance),radius01[:,None].expand_as(distance),distance/R[:,None]),-1)
    assert phi.shape==(len(dd01),side*side,10)and bool(torch.isfinite(phi).all())
    return phi,{'predicted_axis_valid':axis_valid,'disc_axis01':disc,'operational_perpendicular_axis01':perpendicular}

class IoUModel(nn.Module):
    def __init__(self,spatial_dim=3072,global_dim=3072,arm='TEXT28_BASE'):
        super().__init__();spec=setting(arm);self.arm=arm;self.spec=spec;self.side=spec['feature_grid'];self.output_size=spec['heatmap']
        self.spatial_dim=spatial_dim;self.global_dim=global_dim;self.text_mode=spec['text_mode']
        self.local_projection=nn.Linear(spatial_dim,256);self.local_norm=nn.LayerNorm(256)
        self.residual=nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,256))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
        self.global_projection=nn.Linear(global_dim,256);self.global_norm=nn.LayerNorm(256);self.region_norm=nn.LayerNorm(256)
        self.text_projection=nn.Linear(512,256);self.text_norm=nn.LayerNorm(256);self.film=nn.Linear(256,512)
        self.decoder=nn.Sequential(nn.Conv2d(256,128,3,padding=1),nn.GELU(),nn.Conv2d(128,64,3,padding=1),nn.GELU(),nn.Conv2d(64,1,1))
        self.anchor=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,6))
        self.geometry_projection=nn.Linear(10,256);self.geometry_fusion_norm=nn.LayerNorm(256)
        self.register_buffer('prototypes',orthogonal_prototypes())

    def encode_query(self,text_vectors,family_ids=None):
        assert text_vectors.ndim==2 and text_vectors.shape[1]==512
        if self.text_mode=='fixed_normalized_orthogonal_prototypes':
            ids=torch.as_tensor(family_ids,device=text_vectors.device,dtype=torch.long)
            assert ids.shape==(len(text_vectors),)and bool(((ids>=0)&(ids<N_FAMILIES)).all())
            text_vectors=self.prototypes[ids]
        else:assert family_ids is None,'NO_QUERY_FAMILY_ID_IN_REAL_TEXT_FORWARD'
        return F.normalize(self.text_norm(self.text_projection(text_vectors.detach().float())),dim=-1)

    def encode_image(self,local,common_global):
        assert local.shape[1:]==(self.side*self.side,self.spatial_dim)and common_global.shape==(len(local),self.global_dim)
        u=self.local_norm(self.local_projection(local.float()));delta=self.residual(u)
        context=self.global_norm(self.global_projection(common_global.float()));v0=self.region_norm(u+delta+context[:,None])
        values=self.anchor(v0.mean(1)).sigmoid();anchors=values[:,:4].reshape(len(local),2,2);D=D_MAX*values[:,4];R=R_MAX*values[:,5]
        phi,axes=predicted_geometry(anchors,D,R,self.side);v=self.geometry_fusion_norm(v0+self.geometry_projection(phi))
        if self.side==56:
            common=F.avg_pool2d(v.transpose(1,2).reshape(len(v),256,56,56),2,2).flatten(2).transpose(1,2)
        else:common=v
        return {'anchors01':anchors,'dd01':D,'radius01':R,'alignment_features':common,'decoder_features':v,
            'region_features':F.normalize(common,dim=-1),'delta':delta,'geometry_features':phi,**axes}

    def _decode_block(self,v,query):
        gamma,beta=self.film(query).chunk(2,-1);conditioned=v[:,None]*(1+gamma[None,:,None])+beta[None,:,None]
        b,q=conditioned.shape[:2];grid=conditioned.permute(0,1,3,2).reshape(b*q,256,self.side,self.side)
        return F.interpolate(self.decoder(grid),size=(self.output_size,self.output_size),mode='bilinear',align_corners=False).reshape(b,q,self.output_size,self.output_size)

    def decode(self,v,query):
        blocks=[]
        for i in range(0,len(query),3):
            q=query[i:i+3]
            if self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                blocks.append(checkpoint(self._decode_block,v,q,use_reentrant=False,preserve_rng_state=False))
            else:blocks.append(self._decode_block(v,q))
        return torch.cat(blocks,1)

    def forward(self,local,common_global,text_vectors,family_ids=None,valid_support256=None):
        image=self.encode_image(local,common_global);query=self.encode_query(text_vectors,family_ids)
        logits=self.decode(image['decoder_features'],query);p=logits.sigmoid()
        return {**image,'heatmap_logits':logits,'heatmap_probabilities':p,'query_features':query,
            'candidate_scores':score_windows(p,valid_support256)}

    def local_retrieval_scores(self,region_features,text_vectors,support28):
        assert self.text_mode=='shared_frozen_eyeclip'and region_features.shape[1:]==(784,256)
        query=self.encode_query(text_vectors);scores=torch.einsum('btd,qd->bqt',F.normalize(region_features,dim=-1),query)
        return scores.masked_fill(~(support28[:,None]>0),-torch.inf).max(-1).values

def window_support(valid_support256):
    """Image-shape-only support in exactly the frozen784 candidate windows."""
    indices,inside=window_gather_indices();indices=indices.to(valid_support256.device);inside=inside.to(valid_support256.device)
    b=len(valid_support256)
    return ((valid_support256.flatten(1)[:,indices.flatten()].reshape(b,784,-1)>0)&inside[None]).any(-1)

class CrossGalleryHead(nn.Module):
    def __init__(self,spatial_dim=768,global_dim=512):
        super().__init__()
        self.visual=IoUModel(spatial_dim,global_dim,arm='TEXT28_BASE')
        self.primitive_head=nn.Conv2d(256,4,1)

    def encode_text(self,frozen_sentence_vectors):
        assert frozen_sentence_vectors.ndim==2 and frozen_sentence_vectors.shape[1]==512
        return self.visual.encode_query(frozen_sentence_vectors)

    def encode_image(self,local,global_features):
        image=self.visual.encode_image(local,global_features)
        grid=image['decoder_features'].transpose(1,2).reshape(len(local),256,28,28)
        logits=F.interpolate(self.primitive_head(grid),size=(256,256),mode='bilinear',align_corners=False)
        return {**image,'primitive_logits':logits,'primitive_probabilities':logits.float().sigmoid(),
            'native_plus_adapted_features':torch.cat((global_features.float(),image['alignment_features'].mean(1)),-1)}

    def decode(self,image,text_vectors,valid_support256):
        query=self.encode_text(text_vectors);logits=self.visual.decode(image['decoder_features'],query)
        probability=logits.float().sigmoid();candidates=score_windows(probability,valid_support256)
        valid=window_support(valid_support256)
        retrieval=candidates.masked_fill(~valid[:,None],-torch.inf).max(-1).values
        return {'query_features':query,'heatmap_logits':logits,'heatmap_probabilities':probability,
            'candidate_scores':candidates,'retrieval_scores':retrieval,
            'retrieval_score_interface':'MAX_IMAGE_VALID_784_FROZEN_TOP4_WINDOW_MEAN_SIGMOID_QUERY_MAP'}

    def forward(self,local,global_features,text_vectors,valid_support256):
        image=self.encode_image(local,global_features)
        return {**image,**self.decode(image,text_vectors,valid_support256)}

REFINE = CrossGalleryHead

def load_model(path, device='cpu'):
    """Load only a released tensor state, never a research/optimizer pickle."""
    from pathlib import Path
    path=Path(path)
    if path.suffix == '.safetensors':
        from safetensors.torch import load_file
        state=load_file(str(path),device='cpu')
    else:
        state=torch.load(path,map_location='cpu',weights_only=True)
    local_dim=state['visual.local_projection.weight'].shape[1]
    global_dim=state['visual.global_projection.weight'].shape[1]
    model=REFINE(local_dim,global_dim)
    model.load_state_dict(state,strict=True)
    return model.to(device).eval()
