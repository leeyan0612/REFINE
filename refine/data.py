"""Public CaSMO annotation reader; source images/encoders remain caller-owned."""
from functools import lru_cache
from pathlib import Path
import json
import random
import numpy as np
import torch

AXES=('C','S','M')
LESIONS=('EX','HE','MA','SE')
OFFSETS={'C':0,'S':4,'M':44}
ALLOWED={'C':list(range(4)),
         'S':[i for i in range(32)if i not in (6,14,22,30)],
         'M':[i for i in range(20)if i not in (3,8,13,18)]}

def need(value,message):
    if not bool(value):raise ValueError(message)

class AnnotationBank:
    def __init__(self,manifest,feature_root,text_bank,device='cuda'):
        manifest=Path(manifest).resolve()
        # Paths in the manifest are relative to the repository root.
        self.root=manifest.parent.parent
        self.records=sorted(json.loads(manifest.read_text())['records'],
                            key=lambda r:r['source_order_index'])
        self.allowed={k:list(v)for k,v in ALLOWED.items()}
        self.core=[i for i,r in enumerate(self.records)if r['role']=='CORE']
        self.meta=[i for i,r in enumerate(self.records)if r['role']=='META']
        self.cal=[i for i,r in enumerate(self.records)if r['role']=='CAL']
        self.device=device;self.feature_root=Path(feature_root)
        self._member_cache={};self.member_payload_provider=self._member_payload
        self.lesion_ids={a:torch.tensor([q//(1 if a=='C'else 8 if a=='S'else 5)
                          for q in self.allowed[a]],device=device)for a in AXES}
        with np.load(text_bank,allow_pickle=False)as z:
            text=z['train']
            need(text.shape==(64,3,512),'Expected frozen EyeCLIP train text bank [64,3,512]')
            self.train_text=torch.tensor(text,device=device,dtype=torch.float32)
        self.weights={}
        positive={a:np.zeros(len(self.allowed[a]),np.float64)for a in AXES}
        known={a:np.zeros(len(self.allowed[a]),np.float64)for a in AXES}
        for i in self.core:
            z=self._targets(i)
            for axis in AXES:
                ids=self.allowed[axis]
                w=z['M_silver_weight'][ids]if axis=='M'else np.ones(len(ids))
                positive[axis]+=(z[axis+'_positive_mass256'][ids].astype(np.float64)*w[:,None,None]).sum((1,2))
                known[axis]+=(z[axis+'_known_mass256'][ids].astype(np.float64)*w[:,None,None]).sum((1,2))
        for axis in AXES:
            ratio=np.ones_like(positive[axis])
            np.divide(known[axis]-positive[axis],np.maximum(positive[axis],1),out=ratio,where=positive[axis]>0)
            self.weights[axis]=torch.tensor(np.clip(ratio,1,20),device=device,dtype=torch.float32)
        self.weights['primitive']=self.weights['C'].clone()

    @lru_cache(maxsize=2)
    def _targets(self,index):
        ref=self.records[index]['files']['targets']
        with np.load(self.root/ref['file'],allow_pickle=False)as z:
            return {k:z[k].copy()for k in z.files}

    def _member_payload(self,index):
        row=self.records[index]
        groups=json.loads((self.root/row['files']['groups']['file']).read_text())['morphology_groups']
        with np.load(self.root/row['files']['members']['file'],allow_pickle=False)as z:
            sparse={k:z[k].copy()for k in z.files}
        return [g for g in groups if g['query_index']in self.allowed['M']],sparse

    def prepare(self,draw,role):
        permitted=self.core if role=='CORE'else self.meta
        need(set(draw['images'])<=set(permitted),'Content roles must remain disjoint')
        local=[];glob=[];cells=[];targets={a:[]for a in AXES}
        for index,flip in zip(draw['images'],draw['flips']):
            row=self.records[index];z=self._targets(index)
            dataset_dir='IDRiD' if row['dataset']=='IDRID' else 'DDR'
            feature_path=self.feature_root/dataset_dir/(row['public_id']+'.npz')
            with np.load(feature_path,allow_pickle=False)as f:
                # Two independently extracted orientations, not a mirrored token approximation.
                need(f['local'].shape[:2]==(2,784),'Expected both real image orientations')
                local.append(np.asarray(f['local'][int(flip)]).copy())
                glob.append(np.asarray(f['global_features'][int(flip)]).copy())
            A=z['cell_mass256'];A=A[:,::-1].copy()if flip else A.copy();cells.append(A)
            for axis in AXES:
                ids=self.allowed[axis]
                P=z[axis+'_positive_mass256'][ids];K=z[axis+'_known_mass256'][ids]
                if flip:P=P[...,::-1].copy();K=K[...,::-1].copy()
                targets[axis].append(dict(P=P,K=K,truth=z[axis+'_truth'][ids],
                    complete=z[axis+'_complete'][ids],weight=z['M_silver_weight'][ids]
                    if axis=='M'else np.ones(len(ids),np.float32)))
        cells=torch.tensor(np.stack(cells),device=self.device,dtype=torch.float32)
        axes={a:{k:torch.as_tensor(np.stack([v[k]for v in targets[a]]).copy(),device=self.device)
                 for k in targets[a][0]}for a in AXES}
        for a in AXES:
            axes[a]['P']=axes[a]['P'].float();axes[a]['K']=axes[a]['K'].float()
            axes[a]['cells']=cells
        out=dict(images=draw['images'],flips=draw['flips'],
            local=torch.tensor(np.stack(local),device=self.device),
            global_features=torch.tensor(np.stack(glob),device=self.device),
            support=cells>0,axes=axes,lesion_ids=self.lesion_ids,
            query_indices=draw['queries'],query_wording_indices=draw['wordings'],
            positive_weights=self.weights)
        out['vectors']={a:self.train_text[[OFFSETS[a]+q for q in draw['queries'][a]],
                                        draw['wordings'][a]]for a in AXES}
        out['members']=self.members(out,role)
        return out

    def members(self, prepared, role):
            'Construct sparse member masses and group incidence for the specified source role.'
            import numpy as np
            import torch
            need(role in ('CORE', 'META') and set(prepared['images']) <= set(self.core if role == 'CORE' else self.meta),
                 'REFINE_MEMBER_TARGETS_WITHIN_EXPLICIT_ROLE')
            gt = prepared['axes']['M']
            eligible = ((gt['truth'] == 1) & gt['complete'].bool() & (gt['weight'] > 0)
                        & (gt['P'].sum((-2, -1)) > 0)).detach().cpu().numpy()
            images, queries, areas, cells, masses, ptr = [], [], [], [], [], [0]
            group_image, group_query, group_ptr, group_member = [], [], [0], []
            source_groups = skipped = 0
            for image, (record, flip) in enumerate(zip(prepared['images'], prepared['flips'])):
                if record not in self._member_cache:
                    self._member_cache[record] = self.member_payload_provider(record)
                groups, sparse = self._member_cache[record]
                for position, q in enumerate(self.allowed['M']):
                    selected = [g for g in groups if g['query_index'] == q and g['query_complete'] and g['complete_members']]
                    source_groups += len(selected)
                    if not eligible[image, position]:
                        skipped += len(selected)
                        continue
                    unique_members, seen_groups = {}, set()
                    for group in selected:
                        lesion = group['lesion']
                        components, native_areas = group['member_component_ids'], group['native_member_areas']
                        need(lesion == LESIONS[q // 5] and components == group['original_member_component_ids']
                             and components and len(components) == len(native_areas)
                             and len(set(components)) == len(components), 'REFINE_ORIGINAL_UNCROPPED_LEGAL_GROUP')
                        signature = tuple(sorted((lesion, int(x)) for x in components))
                        need(signature not in seen_groups, 'REFINE_NO_DUPLICATE_LEGAL_GROUP_SET')
                        seen_groups.add(signature)
                        incidence = []
                        for component, area in zip(components, native_areas):
                            identity = lesion, int(component)
                            need(type(area) is int and 0 < area < 2 ** 63, 'REFINE_EXACT_POSITIVE_NATIVE_MEMBER_AREA')
                            if identity in unique_members:
                                at = unique_members[identity]
                                need(areas[at] == area, 'REFINE_SHARED_MEMBER_SAME_NATIVE_AREA')
                                incidence.append(at)
                                continue
                            prefix = lesion + '__cell256_'
                            a, b = map(int, sparse[prefix + 'ptr'][int(component):int(component) + 2])
                            index = sparse[prefix + 'index'][a:b].astype(np.int64)
                            mass = sparse[prefix + 'count'][a:b].copy()
                            need(index.ndim == mass.ndim == 1 and index.shape == mass.shape and len(index) > 0
                                 and mass.dtype.kind in 'iu' and (mass > 0).all()
                                 and ((index >= 0) & (index < 65536)).all()
                                 and int(mass.sum(dtype=np.uint64)) == area, 'REFINE_NATIVE_CSR_AREA_CONSERVATION')
                            if flip:
                                index = index // 256 * 256 + 255 - index % 256
                            at = len(areas)
                            unique_members[identity] = at
                            images.append(image); queries.append(position); areas.append(area)
                            cells.append(index); masses.append(mass); ptr.append(ptr[-1] + len(index))
                            incidence.append(at)
                        group_image.append(image); group_query.append(position)
                        group_member.extend(incidence); group_ptr.append(len(group_member))
            arrays = dict(member_image=images, member_query=queries, member_area=areas, member_ptr=ptr,
                          cell_index=np.concatenate(cells) if cells else np.empty(0, np.int64),
                          member_cell_mass=np.concatenate(masses) if masses else np.empty(0, np.int64),
                          group_image=group_image, group_query=group_query,
                          group_ptr=group_ptr, group_member_indices=group_member)
            out = {k: torch.as_tensor(v, device=self.device, dtype=torch.long) for k, v in arrays.items()}
            represented = np.zeros_like(eligible)
            if group_image:
                represented[np.asarray(group_image), np.asarray(group_query)] = True
            need(np.array_equal(represented, eligible), 'REFINE_EVERY_ELIGIBLE_POSITIVE_HAS_ORIGINAL_GROUP')
            out.update(role=role, loss_only=True, members_deduplicated_by_original_component_id=True,
                       original_group_member_association_preserved=True, group_members_uncropped=True,
                       P=gt['P'], K=gt['K'], cells=gt['cells'], truth=gt['truth'], complete=gt['complete'], weight=gt['weight'],
                       positive_mass=gt['P'], known_mass=gt['K'], cell_mass=gt['cells'],
                       image_positive=gt['truth'] == 1, image_known=gt['complete'], silver_weight=gt['weight'],
                       eligibility_counts={k: torch.tensor(v, device=self.device, dtype=torch.long) for k, v in dict(
                           source_complete_groups=source_groups, skipped_ineligible_groups=skipped,
                           retained_complete_groups=len(group_image), shared_unique_member_opportunities=len(areas),
                           retained_group_member_incidence=len(group_member),
                           eligible_complete_positive_image_queries=int(eligible.sum())).items()})
            return out

class Plans:
    'Seeded image, flip and wording streams. Shuffled pools repeat as needed to form batches of sixteen.'
    def __init__(self, seed, pool, allowed, batch_size=16):
        need(type(batch_size) is int and batch_size == 16, 'REFINE_MATCHED_BATCH16')
        need(len(pool) > 0 and len(pool) == len(set(pool)), 'REFINE_NONEMPTY_UNIQUE_DECLARED_ROLE_POOL')
        self.pool, self.allowed, self.pending = list(pool), {a: list(allowed[a]) for a in AXES}, []
        self.batch_size = batch_size
        self.images = random.Random(seed * 1009)
        self.flips = random.Random(seed * 1019)
        self.wording = {a: random.Random(seed * k) for a, k in zip(AXES, (1021, 1031, 1049))}

    def next(self):
        while len(self.pending) < self.batch_size:
            values = self.pool.copy()
            self.images.shuffle(values)
            self.pending.extend(values)
        images, self.pending = self.pending[:self.batch_size], self.pending[self.batch_size:]
        return dict(images=images, flips=[self.flips.randrange(2) for _ in images],
                    queries={a: list(self.allowed[a]) for a in AXES},
                    wordings={a: [self.wording[a].randrange(3) for _ in self.allowed[a]] for a in AXES})
