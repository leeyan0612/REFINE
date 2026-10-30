"""Q-SAT one-step SGD preview, finite-difference meta-update and AdamW fit."""
import contextlib
import types
AXES=('C','S','M')
def need(value,message):
    if not bool(value): raise ValueError(message)

def _public_counts(counts):
    return {k: float(v.detach().cpu()) if v.is_floating_point() else int(v.detach().cpu()) for k,v in counts.items()}
def _clone_function(function, **overrides):
    """Isolated pure-function globals; the source function/module is unchanged."""
    namespace = dict(function.__globals__)
    namespace.update(overrides)
    clone = types.FunctionType(function.__code__, namespace, function.__name__,
                               function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    return clone

def _autocast(device):
    import torch
    return (torch.autocast('cuda', dtype=torch.bfloat16)
            if torch.device(device).type == 'cuda' else contextlib.nullcontext())

def _attach_member_result(result, prepared):
    from .losses import attach_result, member_terms
    allowed = tuple(prepared['query_indices']['M'])
    dynamic_member = _clone_function(member_terms, M_ALLOWED=allowed)

    def member(probability, gt, csr):
        return dynamic_member(probability, gt, csr, m_query_indices=allowed)

    attach = _clone_function(attach_result, member_terms=member)
    return attach(result, prepared, prepared['members'], member_enabled=True,
                  local_enabled=False, supervision_role='CORE')

def step_terms(model, bank, prepared, *, direct_member=True, meta=False):
    from .losses import compute
    with _autocast(bank.device):
        image = model.encode_image(prepared['local'], prepared['global_features'])
        outputs = {a: model.decode(image, prepared['vectors'][a], prepared['support']) for a in AXES}
    result = compute(outputs, image['primitive_logits'], prepared, bank.weights)
    if direct_member and not meta:
        result = _attach_member_result(result, prepared)
    return result

def functional_forward(model, bank, prepared, parameters, *, direct_member=True, meta=False):
    import torch

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.head = model

        def forward(self):
            return step_terms(self.head, bank, prepared, direct_member=direct_member, meta=meta)

    previous = model.visual.training
    model.visual.training = False
    try:
        return torch.func.functional_call(Wrapper(),
            ({'head.' + k: v for k, v in parameters.items()},
             {'head.' + k: v for k, v in model.named_buffers()}), (), strict=True)
    finally:
        model.visual.training = previous

def meta_objective(result, prepared, members, method):
    """Same v34 new-Q function; literal BOTH_OFF uses both switches together."""
    from .satisfaction import new_meta_objective, condition_slots
    need(method in ('NEW_Q', 'NO_CONDITION', 'MEAN_GROUP', 'BOTH_OFF'), 'V35_DEFINED_QUERY_SATISFACTION_VARIANT')
    condition = _clone_function(condition_slots,
        S_ALLOWED=tuple(prepared['query_indices']['S']), M_ALLOWED=tuple(prepared['query_indices']['M']))
    if method == 'BOTH_OFF':
        original_condition = condition

        def condition(probability, batch):
            values, flags, counts = original_condition(probability, batch)
            return values * 0, flags, counts

    objective = _clone_function(new_meta_objective, condition_slots=condition)
    variant = {'NEW_Q': 'SATQ_MEDQ_MEMBER', 'NO_CONDITION': 'SATQ_NO_CONDITION_MEMBER',
               'MEAN_GROUP': 'SATQ_MEAN_GROUP_MEMBER', 'BOTH_OFF': 'SATQ_MEAN_GROUP_MEMBER'}[method]
    return objective(result, prepared, members, variant)

def make_controller(device):
    import torch
    from .losses import AdaptiveWeights, axes_with_M_weights
    from .losses import rebuild_axes

    class Controller(AdaptiveWeights):
        def __init__(self):
            super().__init__(device, medical=True)

        def objective(self, result, prepared, detach=False):
            weights, lesion = self.axis(), self.lesion()
            if detach:
                weights, lesion = weights.detach(), lesion.detach()
            axes = (rebuild_axes(result, prepared, lesion_weights=lesion)
                    if 'member_terms_v31' in result else axes_with_M_weights(result, prepared, lesion))
            return (axes * weights).sum(), weights, axes

    return Controller()

def auto_update(model, bank, controller, result, core, meta, *, method, direct_member=True):
    """Disclosed legacy SGD virtual + mixed FD; actual optimizer is AdamW."""
    import torch
    if not bool(result['axis_defined'].all()):
        return dict(updated=False, reason='MISSING_CORE_AXIS_STATIC_FALLBACK',
                    extra_head_forwards=0, virtual_updates=0, finite_difference_forwards=0, Q_counts=None)
    parameters = dict(model.named_parameters())
    objective, _, _ = controller.objective(result, core, detach=True)
    gradients = torch.autograd.grad(objective, tuple(parameters.values()), retain_graph=True, allow_unused=True)
    virtual = {k: (p - .0003 * ((torch.zeros_like(p) if g is None else g) + .0001 * p)).detach().requires_grad_(True)
               for (k, p), g in zip(parameters.items(), gradients)}
    out = functional_forward(model, bank, meta, virtual, direct_member=direct_member, meta=True)
    target, counts = meta_objective(out, meta, meta['members'], method)
    direction = torch.autograd.grad(target, tuple(virtual.values()), allow_unused=True)
    direction = [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(parameters.values(), direction)]
    norm = float(torch.stack([g.double().square().sum() for g in direction]).sum().sqrt())
    if not math.isfinite(norm) or norm <= 0:
        return dict(updated=False, reason='ZERO_OR_NONFINITE_META_DIRECTION', extra_head_forwards=1,
                    virtual_updates=1, finite_difference_forwards=0, Q_counts=_public_counts(counts))
    epsilon = .01 / norm
    hyper = []
    for sign in (1., -1.):
        params = {k: p.detach() + sign * epsilon * g for (k, p), g in zip(parameters.items(), direction)}
        source = functional_forward(model, bank, core, params, direct_member=direct_member)
        value, _, _ = controller.objective(source, core, detach=False)
        hyper.append(torch.autograd.grad(value, controller.parameters(), allow_unused=True))
    values = [[torch.zeros_like(p) if g is None else g for p, g in zip(controller.parameters(), row)] for row in hyper]
    changed = controller.update(values[0], values[1], .0003 / (2 * epsilon))
    return dict(updated=changed['performed'], reason='UPDATED' if changed['performed'] else 'ZERO_HYPERGRADIENT',
                extra_head_forwards=3, virtual_updates=1, finite_difference_forwards=2,
                objective=float(target.detach()), direction_norm=norm, finite_difference_epsilon=epsilon,
                matched_actual_AdamW=False, Q_counts=_public_counts(counts),
                **{k: v for k, v in changed.items() if k != 'performed'})

def make_optimizer(model):
    import torch
    return torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001,
        betas=(.9,.999),eps=1e-8,foreach=False,fused=False)

def training_step(model, core, optimizer, *, step, controller=None, meta=None,
                  method='NEW_Q'):
    """CORE updates the head. Separate META updates only the controller.

    core/meta are prepared batches with allowed queries, tensors and loss-only
    references. See docs/data_format.md. No references enter model.forward.
    """
    import torch
    from types import SimpleNamespace
    bank=SimpleNamespace(device=str(next(model.parameters()).device),weights=core['positive_weights'])
    model.train();optimizer.zero_grad(set_to_none=True)
    result=step_terms(model,bank,core,direct_member=True)
    meta_details=None
    if controller is not None and step%10==0:
        if meta is None: raise ValueError('A disjoint META batch is required every 10 steps')
        meta_details=auto_update(model,bank,controller,result,core,meta,method=method)
    if controller is not None and bool(result['axis_defined'].all()):
        objective,weights,_=controller.objective(result,core,detach=True)
    else:
        objective=result['axes'].sum();weights=result['axes'].new_ones(3)
    objective.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=False)
    optimizer.step()
    return {'loss':float(objective.detach()),'axis_weights':weights.detach().cpu().tolist(),
            'meta':meta_details}
