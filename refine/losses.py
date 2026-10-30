"""Known-area C/S/M supervision and direct member objectives."""
import math
LESIONS=('EX','HE','MA','SE')
S_ALLOWED=tuple(i for i in range(32) if i not in (6,14,22,30))
M_ALLOWED=tuple(i for i in range(20) if i not in (3,8,13,18))
M_INNER=.1
LOCAL_INNER=.05
AXIS_BASE_SCALE=.1
def masked_query_mean(values, valid, weights=None):
    import torch
    weights = valid.float() if weights is None else weights.float() * valid
    total = weights.sum(0)
    return (values * weights).sum(0) / total.clamp_min(1e-12), total > 0

def image_relevance(scores, positive, negative, weight=None):
    """Positive-set LSE; both known P and known N required per query."""
    import torch
    p, n = positive.bool(), negative.bool()
    weight = torch.ones_like(scores) if weight is None else weight.float()
    eligible = weight > 0; p = p & eligible; n = n & eligible
    s = scores.float().T / .07
    logw = weight.T.clamp_min(1e-12).log()
    flags = p.any(0) & n.any(0)
    den = torch.logsumexp((s + logw).masked_fill(~(p | n).T, -1e4), -1)
    num = torch.logsumexp((s + logw).masked_fill(~p.T, -1e4), -1)
    return torch.where(flags, den - num, torch.zeros_like(den)), flags

def soft_region(probability, P, K, positive, weights=None, negative=None):
    """Positive soft IoU and complete-empty mean activation, not fake IoU."""
    import torch
    p, P, K = probability.float(), P.float(), K.float()
    I = (p * P).sum((-2, -1)); union = (p * K).sum((-2, -1)) + P.sum((-2, -1)) - I
    pv,pf=masked_query_mean(1-I/union.clamp_min(1e-12),positive.bool(),weights)
    if negative is None:return pv,pf
    nv,nf=masked_query_mean((p*K).sum((-2,-1))/K.sum((-2,-1)).clamp_min(1),negative.bool(),weights)
    flags=torch.stack((pf,nf));return(torch.stack((pv,nv))*flags).sum(0)/flags.sum(0).clamp_min(1),flags.any(0)

def lesion_macro(values, defined, lesion_indices, *, enabled=None, lesion_weights=None):
    import torch
    assert values.ndim == defined.ndim == 1 and values.shape == defined.shape
    indices = torch.as_tensor(lesion_indices, device=values.device, dtype=torch.long)
    assert indices.shape == values.shape and bool(((indices >= 0) & (indices < 4)).all())
    flags = defined.bool()
    if enabled is not None:
        enabled = torch.as_tensor(enabled, device=values.device, dtype=torch.bool)
        assert enabled.shape == flags.shape
        flags = flags & enabled
    weights = values.new_ones(4) if lesion_weights is None else lesion_weights
    assert weights.shape == (4,) and bool(torch.isfinite(weights).all()) and bool((weights > 0).all())
    per, available, counts = [], [], []
    for lesion in range(4):
        active = flags & (indices == lesion)
        per.append(torch.where(active, values, torch.zeros_like(values)).sum()/active.sum().clamp_min(1))
        available.append(active.any()); counts.append(active.sum())
    per, available, counts = torch.stack(per), torch.stack(available), torch.stack(counts)
    value = (per*weights).sum()/4
    return value, dict(lesion_values=per, lesion_defined=available,
        lesion_defined_tasks=counts, defined_lesions=available.sum(), all_missing=~available.any(),
        fixed_outer_denominator=4)

def image_mean(value, valid, weight=None):
    import torch
    w = valid.float() if weight is None else weight.float() * valid
    total = w.sum(0)
    return (value * w).sum(0) / total.clamp_min(1e-12), total > 0

def dense_queries(logits, gt, positive_weights):
    """Old native known-area BCE + positive Dice, returning each query."""
    import torch
    from torch.nn import functional as F
    l = logits.float()
    P, K, cells = (gt[k].float() for k in ('P', 'K', 'cells'))
    t, k = P / K.clamp_min(1), K / cells[:, None].clamp_min(1)
    weight = gt['weight'].float()
    mass = k.sum((-2, -1))
    valid = (mass > 0) & (weight > 0)
    positive = (P.sum((-2, -1)) > 0) & valid
    ce = (positive_weights.float()[None, :, None, None] * t * F.softplus(-l)
          + (1-t) * F.softplus(l)) * k
    bce, bf = image_mean(ce.sum((-2, -1))/mass.clamp_min(1e-12), valid, weight)
    p = l.sigmoid()
    inter = (k*t*p).sum((-2, -1)); den = (k*(p+t)).sum((-2, -1))
    dice, _ = image_mean(1-(2*inter+1e-6)/(den+1e-6), positive, weight)
    return bce+dice, bf

def cross_queries(scores, gt):
    """Same positive-set LSE, without inventing missing negatives."""

    return image_relevance(scores, gt['truth'] == 1, gt['truth'] == 0,
                           gt['weight'])

def image_bce_queries(scores, gt):
    from torch.nn import functional as F
    score = scores.float().clamp(1e-6, 1-1e-6)
    target = (gt['truth'] == 1).float()
    return image_mean(F.binary_cross_entropy(score, target, reduction='none'),
                      gt['truth'] >= 0, gt['weight'])

def compute(outputs, primitive_logits, prepared, positive_weights):
    """Return static axis losses plus unweighted M queries for MED CORE only."""
    import torch
    result, flags, counts = [], [], {}
    M_values = M_flags = None
    for axis in ('C', 'S', 'M'):
        out, gt = outputs[axis], prepared['axes'][axis]
        dense, df = dense_queries(out['heatmap_logits'], gt, positive_weights[axis])
        if axis == 'M':
            image, imf = image_bce_queries(out['retrieval_scores'], gt)
            termflags = torch.stack((df, imf))
            values = (torch.stack((dense, image))*termflags).sum(0)/termflags.sum(0).clamp_min(1)
            active = termflags.any(0)
            M_values, M_flags = values, active
            value, info = lesion_macro(values, active, prepared['lesion_ids'][axis])
            result.append(.1*value)
            counts['M_image_BCE_queries'] = imf.sum()
            counts['M_known_image_pairs'] = (gt['truth'] >= 0).sum()
            counts['M_positive_only_query_batches'] = ((gt['truth'] == 1).any(0) & ~(gt['truth'] == 0).any(0)).sum()
        else:
            cross, cf = cross_queries(out['retrieval_scores'], gt)
            # Compose the query task before the class mean. A cross term with
            # no P/N support contributes zero to its own query; it must not
            # acquire extra weight by averaging only cross-supported queries.
            active = df | cf
            per_query = dense*df + (.1 if axis == 'C' else 1.)*cross*cf
            value, info = lesion_macro(per_query, active, prepared['lesion_ids'][axis])
            if axis == 'C':
                primitive, pf = dense_queries(primitive_logits, gt, positive_weights['primitive'])
                pv, _ = lesion_macro(primitive, pf, prepared['lesion_ids'][axis])
                result.append(value+pv)
                counts['C_primitive_queries'] = pf.sum()
            else:
                result.append(.1*value)
            counts[axis+'_cross_queries'] = cf.sum()
            counts[axis+'_cross_no_P'] = ~(gt['truth'] == 1).any(0)
            counts[axis+'_cross_no_N'] = ~(gt['truth'] == 0).any(0)
        flags.append(active.any())
        counts[axis+'_dense_queries'] = df.sum()
        counts[axis+'_defined_queries'] = active.sum()
        counts[axis+'_defined_lesions'] = info['defined_lesions']
        counts[axis+'_unknown_image_pairs'] = (gt['truth'] < 0).sum()
        counts[axis+'_partial_positive_image_pairs'] = ((gt['truth'] == 1) & ~gt['complete']).sum()
    # Vector skip masks are reduced to scalar counters at the source.
    for key in list(counts):
        if counts[key].ndim:
            counts[key] = counts[key].sum()
    return dict(axes=torch.stack(result), axis_defined=torch.stack(flags),
                M_values=M_values, M_defined=M_flags, counts=counts, outputs=outputs)

def axes_with_M_weights(result, prepared, lesion_weights=None):
    import torch
    if lesion_weights is None:
        return result['axes']
    m, _ = lesion_macro(result['M_values'], result['M_defined'],
                         prepared['lesion_ids']['M'], lesion_weights)
    return torch.stack((result['axes'][0], result['axes'][1], .1*m))

class AdaptiveWeights:
    """Three bounded axis controls; four M-only controls in MED_Q."""
    def __init__(self, device, medical=False):
        import torch
        self.medical = medical
        self.floor = torch.tensor([.5, .1, .1] if medical else [.1, .1, .1], device=device)
        initial = (torch.ones(3, device=device)-self.floor)/(3-self.floor.sum())
        self.logits = torch.nn.Parameter(initial.log())
        self.lesion_logits = torch.nn.Parameter(torch.zeros(4, device=device)) if medical else None
        self.optimizer = torch.optim.Adam(self.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0)

    def parameters(self):
        return [self.logits]+([self.lesion_logits] if self.medical else [])

    def axis(self):
        import torch
        return self.floor+(3-self.floor.sum())*torch.softmax(self.logits, -1)

    def lesion(self):
        import torch
        return .25+3*torch.softmax(self.lesion_logits, -1) if self.medical else None

    def objective(self, result, prepared, detach=False):
        weights, lesion = self.axis(), self.lesion()
        if detach:
            weights = weights.detach()
            lesion = None if lesion is None else lesion.detach()
        axes = axes_with_M_weights(result, prepared, lesion)
        return (axes*weights).sum(), weights, axes

    def update(self, positive, negative, scale):
        import torch
        self.optimizer.zero_grad(set_to_none=True)
        before = torch.cat((self.axis().detach(), self.lesion().detach() if self.medical else self.axis().new_empty(0)))
        for p, a, b in zip(self.parameters(), positive, negative):
            p.grad = (-float(scale)*(a-b)).detach()
            assert bool(torch.isfinite(p.grad).all())
        norm = torch.stack([p.grad.double().square().sum() for p in self.parameters()]).sum().sqrt()
        if not bool(norm > 0):
            return dict(performed=False, hypergradient_norm=0., effective_weight_change_norm=0.)
        self.optimizer.step()
        with torch.no_grad():
            self.logits.sub_(self.logits.mean())
            if self.medical:
                self.lesion_logits.sub_(self.lesion_logits.mean())
        after = torch.cat((self.axis().detach(), self.lesion().detach() if self.medical else self.axis().new_empty(0)))
        return dict(performed=True, hypergradient_norm=float(norm),
                    effective_weight_change_norm=float((after-before).norm()))

def need(value, code):
    if not bool(value):
        raise ValueError(code)

def _image_mean(value, valid, weight):
    import torch
    w = torch.where(valid, weight.float(), torch.zeros_like(weight.float()))
    d = w.sum(0)
    return (torch.where(valid, value, torch.zeros_like(value))*w).sum(0)/d.clamp_min(1e-12), d > 0

def _macro(values, flags, lesion_ids, weights=None):

    return lesion_macro(values, flags, lesion_ids, lesion_weights=weights)

def _mass_gt(gt, probability):
    import torch
    P, K, cells = (gt[k].float() for k in ('P', 'K', 'cells'))
    need(P.shape == K.shape == probability.shape and cells.shape == P.shape[:1]+P.shape[2:], 'NATIVE_MASS_AND_READOUT_SHAPES')
    need(torch.isfinite(P).all() and torch.isfinite(K).all() and torch.isfinite(cells).all()
         and (P >= 0).all() and (P <= K).all() and (K <= cells[:, None]).all(), 'POSITIVE_SUBSET_KNOWN_NATIVE_MASS')
    need(gt['truth'].shape == gt['complete'].shape == gt['weight'].shape == P.shape[:2]
         and torch.isfinite(gt['weight']).all() and (gt['weight'] >= 0).all(), 'EXACT_TRUTH_COMPLETENESS_CONFIDENCE_SHAPES')
    return P, K, cells

def member_terms(probability, gt, csr, m_query_indices=M_ALLOWED):
    """Per-query equal-member coverage and complete-positive pollution.

CSR is the v29 native-mass schema. The loader must deduplicate using original
(image, query, lesion, component) before constructing CSR, never coarse-bin
identity. It must not call the old META-only loader for CORE data.
    """
    import torch
    need(list(m_query_indices) == list(M_ALLOWED), 'EXACT_ALLOWED_M16_NO_HELD_MEMBER_TARGETS')
    p = probability.float(); b, q, h, w = p.shape
    need(q == 16 and torch.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'FINITE_M16_PROBABILITIES')
    P, K, cells = _mass_gt(gt, p)
    need(csr.get('members_deduplicated_by_original_component_id') is True, 'SOURCE_NATIVE_MEMBER_ID_DEDUPLICATION_REQUIRED')
    mi, mq, ptr, cell = (csr[k].long() for k in ('member_image', 'member_query', 'member_ptr', 'cell_index'))
    mass, area = csr['member_cell_mass'], csr['member_area']
    size = area.numel()
    need(mi.shape == mq.shape == area.shape == (size,) and ptr.shape == (size+1,)
         and cell.ndim == mass.ndim == 1 and cell.shape == mass.shape
         and ptr[0] == 0 and ptr[-1] == cell.numel() and (ptr[1:] > ptr[:-1]).all(), 'BOUNDED_NONEMPTY_NATIVE_MEMBER_CSR_ROWS')
    need(((mi >= 0) & (mi < b)).all() and ((mq >= 0) & (mq < q)).all()
         and ((cell >= 0) & (cell < h*w)).all() and torch.isfinite(mass).all()
         and torch.isfinite(area).all() and (mass > 0).all() and (area > 0).all(), 'POSITIVE_FINITE_MEMBER_MASS_AND_INDICES')
    eligible = (gt['truth'] == 1) & gt['complete'].bool() & (gt['weight'] > 0) & (P.sum((-2, -1)) > 0)
    need((~eligible[:, :, None, None] | (K == cells[:, None])).all(), 'MEMBER_LOSS_ONLY_FULL_K_COMPLETE_NONEMPTY_REFERENCES')
    count = p.new_zeros(b*q); total = p.new_zeros(b*q)
    minimum = p.new_ones(b*q)
    if size:
        need(eligible[mi, mq].all(), 'CSR_MEMBERS_ONLY_CURRENT_COMPLETE_POSITIVE_QUERY')
        owner = torch.repeat_interleave(torch.arange(size, device=p.device), ptr[1:]-ptr[:-1])
        conserved = torch.zeros(size, device=p.device, dtype=torch.float64).index_add(0, owner, mass.double())
        need(torch.equal(conserved, area.double()), 'EACH_MEMBER_NATIVE_AREA_EXACTLY_CONSERVED')
        value = p.reshape(b*q, h*w)[(mi*q+mq)[owner], cell]*mass.float()
        coverage = p.new_zeros(size).index_add(0, owner, value)/area.float()
        count = count.index_add(0, mi*q+mq, torch.ones_like(coverage))
        total = total.index_add(0, mi*q+mq, 1-coverage)
        minimum = minimum.scatter_reduce(0, mi*q+mq, coverage.detach(), reduce='amin', include_self=True)
    membership = count.reshape(b, q) > 0
    need(torch.equal(membership, eligible), 'EVERY_COMPLETE_POSITIVE_HAS_MEMBERS_NO_SILENT_MISSING_CSR')
    cov, cf = _image_mean((total/count.clamp_min(1)).reshape(b, q), eligible, gt['weight'])
    pollution = (p*(K-P)).sum((-2, -1))/((p*K).sum((-2, -1))+1e-6)
    pol, pf = _image_mean(pollution, eligible, gt['weight'])
    weakest, wf = _image_mean(minimum.reshape(b, q), eligible, gt['weight'])
    need(torch.equal(cf, pf), 'COVERAGE_AND_POLLUTION_SHARE_FIXED_POSITIVE_ELIGIBILITY')
    return dict(coverage=cov, coverage_defined=cf, pollution=pol, pollution_defined=pf,
        weakest_member_diagnostic=weakest.detach(), weakest_member_defined=wf,
        counts=dict(deduplicated_membership_opportunities=p.new_tensor(size, dtype=torch.long),
            eligible_image_query_pairs=eligible.sum(), coverage_queries=cf.sum(), pollution_queries=pf.sum()))

def local_condition_terms(logits, gt, raw_gt, s_query_indices=S_ALLOWED):
    """Direction-balanced local condition loss on the same raw lesion mass.

Only 12 allowed atomic LE1/GT1 pairs. Pa/Pb are native masses, not coarse
subtraction of two masks. Their mass union must equal already supplied raw C.
Mixed 256-bin masses remain fractional; no hard local cell label is invented.
    """
    import torch
    from torch.nn import functional as F
    pairs = condition_pairs(s_query_indices)
    l = logits.float(); need(l.ndim == 4 and l.shape[1] == 28 and torch.isfinite(l).all(), 'FINITE_ALLOWED_S28_LOGITS')
    P, K, cells = _mass_gt(gt, l.sigmoid())
    raw, rawK, rawcells = _mass_gt(raw_gt, l[:, :4])
    need(raw.shape == (l.shape[0], 4, *l.shape[-2:]) and torch.equal(rawcells, cells), 'SAME_IMAGE_RAW_C4_NATIVE_MASS')
    values, flags, image_flags, directions = [], [], [], []
    for pair in pairs:
        a, b, lesion = pair['left_column'], pair['right_column'], pair['lesion_index']
        eligible = (gt['complete'][:, a] & gt['complete'][:, b] & raw_gt['complete'][:, lesion]
            & (gt['truth'][:, a] >= 0) & (gt['truth'][:, b] >= 0) & (raw_gt['truth'][:, lesion] >= 0)
            & (gt['weight'][:, a] > 0) & (gt['weight'][:, b] > 0) & (raw_gt['weight'][:, lesion] > 0))
        Pa, Pb = P[:, a], P[:, b]
        need((~eligible[:, None, None] | ((Pa+Pb == raw[:, lesion]) & (K[:, a] == cells)
             & (K[:, b] == cells) & (rawK[:, lesion] == cells))).all(), 'ALLOWED_ATOMIC_PAIR_PARTITIONS_SAME_COMPLETE_RAW_LESION')
        weight = torch.minimum(torch.minimum(gt['weight'][:, a], gt['weight'][:, b]), raw_gt['weight'][:, lesion])
        delta = l[:, a]-l[:, b]
        A, B = Pa.sum((-2, -1)), Pb.sum((-2, -1))
        fa, fb = eligible & (A > 0), eligible & (B > 0)
        va = (F.softplus(-delta)*Pa).sum((-2, -1))/A.clamp_min(1)
        vb = (F.softplus(delta)*Pb).sum((-2, -1))/B.clamp_min(1)
        directions.append(fa.sum()+fb.sum())
        # Each defined local direction gets one half when both are present.
        per = (va*fa+vb*fb)/(fa.float()+fb.float()).clamp_min(1)
        value, flag = _image_mean(per[:, None], (fa | fb)[:, None], weight[:, None])
        values.append(value[0]); flags.append(flag[0]); image_flags.append((fa | fb).sum())
    values, flags = torch.stack(values), torch.stack(flags)
    ids = [x['lesion_index'] for x in pairs]
    loss, info = _macro(values, flags, ids)
    return dict(value=loss, pair_values=values, pair_defined=flags, lesion_info=info,
        counts=dict(local_pairs=flags.sum(), local_image_pair_opportunities=torch.stack(image_flags).sum(),
            defined_local_directions=torch.stack(directions).sum()), pair_specification=pairs)

def attach_result(result, prepared, csr, *, member_enabled=True, local_enabled=False, supervision_role='CORE'):
    """Keep old axes/M_values interface while retaining all original components.

Call only on CORE/FIT optimizer batches, including the CORE side of the
virtual/FD recipe. META uses the original unweighted base+0.1Q target: Q
already had member coverage/pollution in v29, so it is not added again here.
    """
    import torch
    need(supervision_role in ('CORE', 'FIT'), 'DIRECT_LOSS_IS_NOT_THE_META_QUALITY_TARGET')
    need('member_terms_v31' not in result, 'NO_DOUBLE_ATTACHMENT_OF_DIRECT_MEMBER_LOSS')
    ids = prepared['lesion_ids']['M']
    members = member_terms(result['outputs']['M']['heatmap_probabilities'], prepared['axes']['M'], csr)
    need((~members['coverage_defined'] | result['M_defined']).all(), 'NEW_MEMBER_QUALIFICATION_SUBSET_OF_ORIGINAL_M_DEFINED_QUERIES')
    local = (local_condition_terms(result['outputs']['S']['heatmap_logits'], prepared['axes']['S'], prepared['axes']['C'])
             if local_enabled else None)
    out = dict(result, axes_original=result['axes'], M_values_original=result['M_values'],
        member_terms_v31=members, local_terms_v31=local, member_enabled_v31=bool(member_enabled))
    counts = dict(result['counts'])
    counts.update({'V31_'+key: value for key, value in members['counts'].items()})
    if local is not None:
        counts.update({'V31_'+key: value for key, value in local['counts'].items()})
    out['counts'] = counts
    addition = .5*M_INNER*(members['coverage']+members['pollution']) if member_enabled else result['M_values']*0
    out['M_values'] = result['M_values']+addition
    m, _ = _macro(out['M_values'], result['M_defined'], ids)
    s = result['axes'][1]+(AXIS_BASE_SCALE*LOCAL_INNER*local['value'] if local is not None else result['axes'][1]*0)
    out['axes'] = torch.stack((result['axes'][0], s, AXIS_BASE_SCALE*m))
    return out

def rebuild_axes(result, prepared, *, lesion_weights=None, member_multipliers=None):
    """CORE objective only: original C/S/M plus two shared member components.

The two multipliers are class-shared; optional original M lesion weights
still act on all M terms within their fixed /4. No hidden active-axis mean.
    """
    import torch
    need('member_terms_v31' in result, 'DIRECT_MEMBER_DECOMPOSITION_PRESENT')
    w = result['axes'].new_ones(2) if member_multipliers is None else member_multipliers
    need(w.shape == (2,) and torch.isfinite(w).all() and (w >= .25).all()
         and torch.isclose(w.sum(), w.new_tensor(2.), atol=1e-6, rtol=0), 'TWO_SHARED_MULTIPLIERS_FLOOR_QUARTER_SUM_TWO')
    g = result['member_terms_v31']
    value = result['M_values_original']
    if result['member_enabled_v31']:
        value = value+.5*M_INNER*(w[0]*g['coverage']+w[1]*g['pollution'])
    m, _ = _macro(value, result['M_defined'], prepared['lesion_ids']['M'], lesion_weights)
    return torch.stack((result['axes_original'][0], result['axes'][1], AXIS_BASE_SCALE*m))

def condition_pairs(s_query_indices):
    """12 fixed native complements, never the held AND_TOWARD complement."""
    ids = list(s_query_indices)
    need(ids == list(S_ALLOWED), 'EXACT_FROZEN_ALLOWED_S28_QUERY_ORDER')
    rows = []
    for lesion in range(4):
        for family, offset in (('PIXEL', 0), ('COMPONENT_BOUNDARY', 2), ('COMPONENT_CENTROID', 4)):
            a, b = lesion*8+offset, lesion*8+offset+1
            rows.append(dict(lesion_index=lesion, operator=family,
                left_query_index=a, right_query_index=b,
                left_column=ids.index(a), right_column=ids.index(b)))
    return rows
