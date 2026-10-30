"""Image retrieval AP with exact-tie grouping and unknown-reference exclusion."""
import math

def image_ap(scores,relevance):
    scores=[float(x)for x in scores];relevance=[int(x)for x in relevance]
    if len(scores)!=len(relevance)or not all(math.isfinite(x)for x in scores):
        raise ValueError('Finite scores and one reference per gallery image are required')
    if any(y not in (-1,0,1)for y in relevance):raise ValueError('References must be -1/0/1')
    order=sorted((i for i,y in enumerate(relevance)if y>=0),key=lambda i:(-scores[i],i))
    denominator=sum(y==1 for y in relevance)
    if not denominator:return None
    total=0.;cumulative=0;i=0
    while i<len(order):
        j=i+1
        while j<len(order)and scores[order[j]]==scores[order[i]]:j+=1
        good=sum(relevance[k]==1 for k in order[i:j]);cumulative+=good
        total+=good*cumulative/j;i=j
    return total/denominator

def image_map(scores,relevance):
    """Return per-query AP; a strict macro is undefined if any query has no positive."""
    values=[image_ap(s,y)for s,y in zip(scores,relevance)]
    return {'per_query_ap':values,'strict_map':None if not values or any(v is None for v in values)
            else math.fsum(values)/len(values),'defined_queries':sum(v is not None for v in values)}
