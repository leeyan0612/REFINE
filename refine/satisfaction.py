"""v34 unweighted META query satisfaction; pure lazy-Torch loss helpers.

No model/authority/I/O/optimizer is defined here. The original CORE objective,
seven controller logits, M_ONLY scope, true AdamW and LEGACY_SGD_FD preview
remain caller-owned and unchanged. Geometry/member/query identifiers are loss
targets only; no target-side field is accepted by a neural forward here.
"""
import math

LESIONS = ('EX', 'HE', 'MA', 'SE')
S_ALLOWED = tuple(i for i in range(32) if i not in (6, 14, 22, 30))
M_ALLOWED = tuple(i for i in range(20) if i not in (3, 8, 13, 18))
VARIANTS = ('SATQ_MEDQ_MEMBER', 'SATQ_NO_CONDITION_MEMBER', 'SATQ_MEAN_GROUP_MEMBER')


def need(value, code):
    if not bool(value):
        raise ValueError(code)


def contract():
    return dict(schema='V34_FIXED_QUERY_SATISFACTION_META_CONTRACT_V1',
        member_temperature=.05, group_temperature=.05, condition_margin=.1, condition_temperature=.1,
        condition_coefficient=1., quality_coefficient=.1,
        group_aggregation='NORMALIZED_LOG_MEAN_EXP_MAX_GROUP_OF_NORMALIZED_LOG_MEAN_EXP_MIN_MEMBER',
        mean_ablation='ORIGINAL_FLAT_UNIQUE_MEMBER_MEAN_CHANGES_WHOLE_GROUP_AGGREGATION_NOT_ONLY_MIN',
        S_condition_reduction='ADDITIVE_TO_UNCHANGED_OLD_PER_QUERY_BASE_NO_RENORMALIZATION',
        no_condition='GRAPH_ZERO_SIGNAL_SAME_SOURCE_ELIGIBILITY_AND_BASE_DENOMINATORS',
        META_learned_weight_multipliers=False, reference_eligibility_model_independent=True,
        query_then_lesion_fixed_four_then_available_axes=True,
        hard_E2E_rule_unchanged=True, soft_group_score_is_not_hard_E2E=True)


def _dtype(value):
    import torch
    return value if value.dtype == torch.float64 else value.float()


def _mean(value, valid, weight):
    import torch
    w = torch.where(valid, weight.to(value.dtype), torch.zeros_like(weight, dtype=value.dtype))
    d = w.sum(0)
    return (torch.where(valid, value, torch.zeros_like(value))*w).sum(0)/d.clamp_min(1e-12), d > 0


def group_scores(probability, members, *, member_temperature=.05, group_temperature=.05):
    """Smooth existential whole-group coverage and original flat-mean companion.

    Supports FP64 for finite synthetic derivative checks; production inputs are
    the existing FP32 probability maps. Hsoft is bounded/monotone but differs
    from the hard max/min of binary-thresholded native coverage in evaluation.
    """
    import torch
    p = _dtype(probability); b, q, h, w = p.shape
    need(torch.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'FINITE_PROBABILITY_MAP_NOT_LOGIT_OR_HARD_TARGET')
    need(member_temperature == .05 and group_temperature == .05, 'FIXED_PREFIT_TEMPERATURES_NOT_TARGET_SCORE_SEARCH')
    need(members.get('members_deduplicated_by_original_component_id') is True
         and members.get('original_group_member_association_preserved') is True
         and members.get('group_members_uncropped') is True, 'SOURCE_ORIGINAL_GROUP_INCIDENCE_REQUIRED')
    mi, mq, ptr, cell = (members[key].long() for key in ('member_image','member_query','member_ptr','cell_index'))
    area, mass = members['member_area'].to(p.dtype), members['member_cell_mass'].to(p.dtype)
    gi, gq, gp, gm = (members[key].long() for key in ('group_image','group_query','group_ptr','group_member_indices'))
    size, groups = area.numel(), gi.numel()
    need(mi.shape == mq.shape == area.shape == (size,) and ptr.shape == (size+1,)
         and cell.ndim == mass.ndim == 1 and cell.shape == mass.shape and ptr[0] == 0
         and ptr[-1] == cell.numel() and (ptr[1:] > ptr[:-1]).all(), 'EXACT_NONEMPTY_MEMBER_CSR_ROWS')
    need(gi.shape == gq.shape == (groups,) and gp.shape == (groups+1,) and gm.ndim == 1
         and gp[0] == 0 and gp[-1] == gm.numel() and (gp[1:] > gp[:-1]).all(), 'EXACT_NONEMPTY_ORIGINAL_GROUP_INCIDENCE')
    need(((mi >= 0) & (mi < b)).all() and ((mq >= 0) & (mq < q)).all()
         and ((gi >= 0) & (gi < b)).all() and ((gq >= 0) & (gq < q)).all()
         and ((cell >= 0) & (cell < h*w)).all() and ((gm >= 0) & (gm < size)).all()
         and torch.isfinite(area).all() and torch.isfinite(mass).all() and (area > 0).all() and (mass > 0).all(),
         'FINITE_ORIGINAL_GROUP_MEMBER_OWNERS_AND_NATIVE_MASS')
    anchor = p.sum((-2,-1))*0
    if size:
        owner = torch.repeat_interleave(torch.arange(size, device=p.device), ptr[1:]-ptr[:-1])
        conserved = torch.zeros(size, device=p.device, dtype=torch.float64).index_add(0, owner, members['member_cell_mass'].double())
        need(torch.equal(conserved, members['member_area'].double()), 'NATIVE_MEMBER_AREA_CONSERVED_EXACTLY')
        weighted = p.reshape(b*q,h*w)[(mi*q+mq)[owner],cell]*mass
        coverage = p.new_zeros(size).index_add(0,owner,weighted)/area
    else:
        coverage = p.reshape(-1)[:0]
    pointers = gp.detach().cpu().tolist()
    group_soft, group_hard = [], []
    for g in range(groups):
        indices = gm[pointers[g]:pointers[g+1]]
        need(indices.numel() > 0 and indices.unique().numel() == indices.numel()
             and (mi[indices] == gi[g]).all() and (mq[indices] == gq[g]).all(), 'ONE_WHOLE_GROUP_SAME_IMAGE_AND_QUERY')
        values = coverage[indices]
        group_soft.append(values[0] if indices.numel() == 1 else
            -member_temperature*(torch.logsumexp(-values/member_temperature,0)-math.log(indices.numel())))
        group_hard.append(values.detach().min())
    soft = torch.stack(group_soft) if group_soft else coverage
    hard = torch.stack(group_hard) if group_hard else coverage.detach()
    hs, hh, flat, flags = [], [], [], []
    for i in range(b):
        for j in range(q):
            gmask, mmask = (gi == i) & (gq == j), (mi == i) & (mq == j)
            n = int(gmask.sum().item())
            need(bool(mmask.any()) == bool(n), 'GROUP_INCIDENCE_AND_FLAT_MEMBER_UNION_HAVE_SAME_OWNERS')
            if n:
                hs.append(soft[gmask][0] if n == 1 else
                    group_temperature*(torch.logsumexp(soft[gmask]/group_temperature,0)-math.log(n)))
                hh.append(hard[gmask].max())
                flat.append(coverage[mmask].mean()); flags.append(True)
            else:
                hs.append(anchor[i,j]); hh.append(anchor[i,j].detach()); flat.append(anchor[i,j]); flags.append(False)
    hsoft = torch.stack(hs).reshape(b,q)
    hard_on_soft = torch.stack(hh).reshape(b,q)
    flat_mean = torch.stack(flat).reshape(b,q)
    valid = torch.tensor(flags, device=p.device, dtype=torch.bool).reshape(b,q)
    need(torch.isfinite(hsoft).all() and ((hsoft >= -1e-6) & (hsoft <= 1+1e-6)).all(), 'BOUNDED_MONOTONE_SOFT_GROUP_PROXY_NOT_HARD_SUCCESS')
    return dict(H_soft=hsoft, H_hard_on_soft_coverage=hard_on_soft, flat_unique_member_mean=flat_mean,
        defined=valid, unique_member_coverage=coverage,
        counts=dict(Q_M_legal_groups=p.new_tensor(groups,dtype=torch.long),
            Q_M_unique_members=p.new_tensor(size,dtype=torch.long),
            Q_M_group_image_query_opportunities=valid.sum(),
            Q_M_soft_hard_coverage_proxy_max_abs_difference=(hsoft-hard_on_soft).abs().masked_fill(~valid,0).max().detach()))


def condition_slots(probability, prepared):
    """Twelve legal atomic pairs; same wording event, both exclusive ROIs P.

    Native fractional masses are retained. A mixed prediction cell may contain
    mass from both disjoint native regions: Pa*Pb==0 is deliberately NOT used.
    """
    import torch
    from torch.nn import functional as F
    p = _dtype(probability); b,q,h,w = p.shape
    need(q == 28 and prepared['query_indices']['S'] == list(S_ALLOWED)
         and prepared['query_indices']['M'] == list(M_ALLOWED)
         and prepared['query_indices']['C'] == list(range(4)), 'ONLY_ORIGINAL_ALLOWED_PRIMARY_QUERIES_NOT_HELD_OR_AUX')
    wordings = prepared['query_wording_indices']['S']
    need(len(wordings) == 28 and all(type(x) is int and 0 <= x < 3 for x in wordings), 'ORIGINAL_SAME_EVENT_THREE_WORDING_PLAN_IDS')
    gt, raw = prepared['axes']['S'], prepared['axes']['C']
    P,K,cells = (gt[key].to(p.dtype) for key in ('P','K','cells'))
    rawP,rawK = raw['P'].to(p.dtype), raw['K'].to(p.dtype)
    need(P.shape == K.shape == p.shape and rawP.shape == rawK.shape == (b,4,h,w)
         and cells.shape == (b,h,w) and torch.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
         and torch.isfinite(P).all() and torch.isfinite(K).all() and (P >= 0).all() and (P <= K).all(),
         'SAME_IMAGE_NATIVE_PROBABILITIES_AND_SOURCE_REFERENCE_MASS')
    values, defined = {}, {}
    matched = eligible_images = 0
    for lesion in range(4):
        for offset in (0,2,4):
            qa,qb = lesion*8+offset, lesion*8+offset+1
            a,bb = S_ALLOWED.index(qa), S_ALLOWED.index(qb)
            same_wording = wordings[a] == wordings[bb]
            matched += int(same_wording)
            Pa,Pb = P[:,a],P[:,bb]
            A,B = Pa.sum((-2,-1)),Pb.sum((-2,-1))
            eligible = (gt['complete'][:,a].bool() & gt['complete'][:,bb].bool() & raw['complete'][:,lesion].bool()
                & (gt['truth'][:,a] == 1) & (gt['truth'][:,bb] == 1) & (raw['truth'][:,lesion] == 1)
                & (gt['weight'][:,a] > 0) & (gt['weight'][:,bb] > 0) & (raw['weight'][:,lesion] > 0)
                & (A > 0) & (B > 0))
            need((~eligible[:,None,None] | ((Pa+Pb == rawP[:,lesion]) & (K[:,a] == cells)
                 & (K[:,bb] == cells) & (rawK[:,lesion] == cells))).all(), 'FROZEN_NATIVE_ATOMIC_PARTITION_AND_COMPLETE_K_NOT_COARSE_LABELS')
            eligible = eligible & same_wording
            eligible_images += int(eligible.sum().item())
            delta = p[:,a]-p[:,bb]
            left = (.1*F.softplus((.1-delta)/.1)*Pa).sum((-2,-1))/A.clamp_min(1)
            right = (.1*F.softplus((.1+delta)/.1)*Pb).sum((-2,-1))/B.clamp_min(1)
            symmetric = .5*(left+right)
            weight = torch.minimum(torch.minimum(gt['weight'][:,a],gt['weight'][:,bb]),raw['weight'][:,lesion])
            value, flag = _mean(symmetric[:,None],eligible[:,None],weight[:,None])
            for column in (a,bb):
                values[column],defined[column] = value[0],flag[0]
    zero = p.sum()*0
    v = torch.stack([values.get(i,zero) for i in range(28)])
    f = torch.stack([defined.get(i,torch.zeros((),device=p.device,dtype=torch.bool)) for i in range(28)])
    return v,f,dict(Q_S_condition_legal_pairs=p.new_tensor(12,dtype=torch.long),
        Q_S_condition_wording_matched_pairs=p.new_tensor(matched,dtype=torch.long),
        Q_S_condition_eligible_image_pairs=p.new_tensor(eligible_images,dtype=torch.long),
        Q_S_condition_defined_query_slots=f.sum(),Q_S_condition_unqualified_query_slots=(~f).sum())


def new_meta_objective(result, prepared, members, variant, cfg=None):
    """Unweighted old base + .1 new Q; no learned lambda/mu/nu argument.

    C stays original. S adds one condition signal to its unchanged old per-query
    quality base. M retains its old four slots, replacing only member feedback.
    NO_CONDITION zeros that additive signal without changing any denominator.
    """
    import torch
    from .losses import image_relevance, soft_region
    from .losses import lesion_macro
    need(variant in VARIANTS and (cfg is None or cfg == contract()), 'EXACT_PREFIT_Q_VARIANT_AND_FIXED_CONTRACT')
    need(members.get('role') == 'META' and members.get('loss_only') is True
         and 'member_terms_v31' not in result, 'META_ORIGINAL_UNWEIGHTED_BASE_NOT_DIRECT_CORE_MEMBER_ATTACHMENT')
    values, flags, counts = [],[],{}
    for axis in ('C','S','M'):
        out,gt = result['outputs'][axis],prepared['axes'][axis]
        pos,neg = gt['truth'] == 1,gt['truth'] == 0
        retrieval,rf = image_relevance(out['retrieval_scores'],pos,neg,gt['weight'])
        region,sf = soft_region(out['heatmap_probabilities'],gt['P'],gt['K'],pos,gt['weight'],neg)
        t,f = torch.stack((retrieval,region)),torch.stack((rf,sf))
        if axis == 'M':
            group = group_scores(out['heatmap_probabilities'],members)
            eligible = pos & gt['complete'].bool() & (gt['weight'] > 0) & (gt['P'].sum((-2,-1)) > 0)
            need(torch.equal(group['defined'],eligible), 'MODEL_INDEPENDENT_GROUP_QUALIFICATION_COMPLETE_POSITIVE_ONLY')
            p,P,K = (out['heatmap_probabilities'].float(),gt['P'].float(),gt['K'].float())
            need((~eligible[:,:,None,None] | (K == gt['cells'][:,None])).all(), 'FULL_K_NATIVE_REFERENCE_FOR_GROUP_AND_POLLUTION')
            h = group['flat_unique_member_mean'] if variant == 'SATQ_MEAN_GROUP_MEMBER' else group['H_soft']
            coverage,cf = _mean(1-h,eligible,gt['weight'])
            pollution = (p*(K-P)).sum((-2,-1))/((p*K).sum((-2,-1))+1e-6)
            pol,pf = _mean(pollution,eligible,gt['weight'])
            need(torch.equal(cf,pf), 'SAME_FIXED_COMPLETE_POSITIVE_GROUP_AND_POLLUTION_ELIGIBILITY')
            t,f = torch.stack((retrieval,region,coverage,pol)),torch.stack((rf,sf,cf,pf))
            counts.update(group['counts']);counts.update(Q_M_member_queries=cf.sum(),Q_M_contamination_queries=pf.sum())
        per = (t*f).sum(0)/f.sum(0).clamp_min(1)
        active = f.any(0)
        if axis == 'S':
            cond,condf,extra = condition_slots(out['heatmap_probabilities'],prepared)
            need((~condf | active).all(), 'CONDITION_SIGNAL_NEVER_CREATES_A_NEW_BASE_QUERY_PERMISSION')
            per = per+(cond*0 if variant == 'SATQ_NO_CONDITION_MEMBER' else cond)
            counts.update(extra)
        value,info = lesion_macro(per,active,prepared['lesion_ids'][axis])
        values.append(value);flags.append(active.any())
        counts['Q_'+axis+'_retrieval_queries']=rf.sum();counts['Q_'+axis+'_region_queries']=sf.sum()
        counts['Q_'+axis+'_defined_lesions']=info['defined_lesions']
    values,flags = torch.stack(values),torch.stack(flags)
    quality = (values*flags).sum()/flags.sum().clamp_min(1)
    baseflags = result['axis_defined']
    base = (result['axes']*baseflags).sum()/baseflags.sum().clamp_min(1)
    counts.update(Q_defined_axes=flags.sum(),meta_defined_axes=baseflags.sum(),Q_defined=flags.any())
    return base+.1*quality,counts


meta_objective = new_meta_objective
