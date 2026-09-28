"""Scoring, calibration and paired metrics for the current seven-detector RAID study."""
from __future__ import annotations
import csv, math, os, random
from collections import Counter, defaultdict
import numpy as np
from experiment_core.analysis.evaluation import actual_fpr, auroc, calibration_threshold, percentile_interval
from experiment_core.detectors.scoring import EPS
from experiment_core.detectors.detector_revision import (
    ORIGIN_NLL, ORIGIN_DENOMINATOR, ORIGIN_NUMERATOR, origin_components, origin_score,
    constant_clean_scores, require_valid_lrr_clipping_spec, structurally_constant_candidate,
    valid_lrr_clipping_spec)
from RAID.raid_data import measure_realized_contamination

DETECTORS = ('log_likelihood', 'rank', 'log_rank', 'lrr', 'entropy', 'entropy_gap', 'binocular_origin')
QUANTILE_GRID = (0.80, 0.85, 0.90, 0.95, 0.975, 0.99, 0.995)
TARGET_FPR = .05
FEATURES = {
 "logp":("logp","log_probability","token_logp"), "rank":("rank","token_rank"),
 "log_rank":("log_rank","token_log_rank"), "entropy":("entropy","token_entropy"),
 "performer_nll":("performer_nll","binoculars_performer_nll"),
 "cross_entropy":("observer_to_performer_cross_entropy","cross_entropy","binoculars_cross_entropy"),
 "token_ids":("scored_token_ids","token_ids"),
 "full_token_ids":("full_scored_token_ids",),
}


def _first(r,names,default=None):
    return next((r[n] for n in names if n in r and r[n] is not None),default)

def source(r): return str(_first(r,("source_id","sample_id","human_source_id")))

def domain(r): return str(_first(r,("domain","dataset_domain","genre")))

def split(r):
    v=str(_first(r,("split","partition"),""))
    return {"tuning":"clipping_tuning","tune":"clipping_tuning"}.get(v,v)

def human(r): return str(_first(r,("label","origin","text_origin"),"")).lower() in {"human","h","0"}

def condition(r):
    if human(r): return "human"
    v=str(_first(r,("attack","condition","attack_name","contamination_mode"),"none")).lower()
    v=v.replace("-","_").replace(" ","_")
    return {"clean":"none","original":"none","unattacked":"none","no_attack":"none",
      "delete_article":"article_deletion","delete_articles":"article_deletion",
      "synonyms":"synonym","homoglyphs":"homoglyph",
      "misspelling":"perplexity_misspelling","misspellings":"perplexity_misspelling",
      "paraphrasing":"paraphrase",
      "whitespace_insertion":"whitespace"}.get(v,v)

def feat(r,name,dtype=float):
    fs=r.get("token_features",r.get("features",{}))
    for alias in FEATURES[name]:
        if alias in fs:
            x=np.asarray(fs[alias],dtype=dtype)
            if x.ndim!=1 or not len(x): raise ValueError(f"empty/invalid feature {alias}")
            return x
    raise KeyError(f"missing token feature {name}")

def local(r,d):
    if d=="log_likelihood": return feat(r,"logp")
    if d=="rank": return feat(r,"rank")
    if d=="log_rank": return feat(r,"log_rank")
    if d=="entropy": return feat(r,"entropy")
    if d=="entropy_gap": return -feat(r,"logp")-feat(r,"entropy")
    raise KeyError(d)

def raw_document_score(r,d):
    if d=="binocular_origin": return origin_score(r)
    ds=r.get("doc_scores",{})
    if d in ds: return float(ds[d])
    if d=="lrr": return float((-feat(r,"logp")).mean()/(feat(r,"log_rank").mean()+EPS))
    return float(local(r,d).mean())

def oriented_document_score(r,d,direction,spec=None):
    spec=spec or {}
    if not spec: return direction*raw_document_score(r,d)
    if d=="binocular_origin": return direction*origin_score(r,spec)
    if d=="lrr":
        require_valid_lrr_clipping_spec(spec)
        n=np.minimum(-feat(r,"logp"),spec["nll_upper"])
        q=np.minimum(feat(r,"log_rank"),spec["log_rank_upper"])
        return direction*float(n.mean()/(q.mean()+EPS))
    m=float(np.maximum(direction*local(r,d),spec["lower"]).mean())
    return m

def stable_row_key(r):
    explicit=_first(r,("row_key","stable_row_key"))
    if explicit is not None:return (str(explicit),)
    return (source(r),split(r),"human" if human(r) else "machine",condition(r),
            str(_first(r,("base_generation_id","generation_id","machine_id"),"")))

def merge_score_rows(falcon_rows,binoculars_rows=None):
    out={stable_row_key(r):{**r,"doc_scores":dict(r.get("doc_scores",{})),
        "token_features":dict(r.get("token_features",{}))} for r in falcon_rows}
    if len(out)!=len(falcon_rows): raise ValueError("duplicate Falcon row key")
    if binoculars_rows is None:return list(out.values())
    b={stable_row_key(r):r for r in binoculars_rows}
    if len(b)!=len(binoculars_rows):raise ValueError("duplicate Binoculars row key")
    if set(out)!=set(b): raise ValueError("Falcon/Binoculars row-key mismatch")
    for k,br in b.items():
        out[k].setdefault("doc_scores",{}).update({
            x:y for x,y in br.get("doc_scores",{}).items()
            if x in {"binoculars", "binocular_origin", ORIGIN_NUMERATOR,
                     ORIGIN_DENOMINATOR}
        })
        names=set(FEATURES["performer_nll"]+FEATURES["cross_entropy"])|{ORIGIN_NLL}
        out[k].setdefault("token_features",{}).update({x:y for x,y in br.get("token_features",{}).items() if x in names})
    return list(out.values())

def token_edit_counts(original, attacked):
    """Compatibility wrapper over the shared, prefix-trimmed RapidFuzz path."""
    z=measure_realized_contamination(original,attacked)
    return {"substitutions":z["substitution_count"],"deletions":z["deletion_count"],
      "insertions":z["insertion_count"],"edit_distance":z["levenshtein_distance"],
      "contamination_rate":z["realized_contamination_rate"]}

def attach_contamination_rates(rows):
    none={(split(r),source(r)):r for r in rows if not human(r) and condition(r)=="none"}
    for r in rows:
        if human(r):continue
        base=r if condition(r)=="none" else none.get((split(r),source(r)))
        if base is None: raise ValueError(f"missing none row for {source(r)}")
        kwargs={}
        base_features=base.get("token_features",{}); attacked_features=r.get("token_features",{})
        full_name=FEATURES["full_token_ids"][0]
        if full_name in base_features and full_name in attacked_features:
            kwargs={"clean_full_token_ids":base_features[full_name],
                    "attacked_full_token_ids":attacked_features[full_name]}
        measured=measure_realized_contamination(
          feat(base,"token_ids",np.int64),feat(r,"token_ids",np.int64),**kwargs)
        stats={**measured,"substitutions":measured["substitution_count"],"deletions":measured["deletion_count"],
          "insertions":measured["insertion_count"],"edit_distance":measured["levenshtein_distance"],
          "contamination_rate":measured["realized_contamination_rate"]}
        for k,v in stats.items():
            if r.get(k) is not None and v is not None and not math.isclose(float(r[k]),float(v),rel_tol=0,abs_tol=1e-12):
                raise ValueError(f"saved {k} disagrees with scored-token alignment")
            r[k]=v
    return rows

def rho(r): return float(r.get("contamination_rate",0.0))

def learn_direction(d,h,m,revised=False):
    if d=="binocular_origin":return -1
    if d=="lrr":return 1
    return 1 if np.mean([raw_document_score(r,d) for r in m])>=np.mean([raw_document_score(r,d) for r in h]) else -1

def candidate_specifications(d,direction,clean,quantiles=QUANTILE_GRID):
    out=[{}]
    if d=="binocular_origin":
        a=np.concatenate([origin_components(r)[0] for r in clean])
        return out+[{"nll_upper":float(np.quantile(a,q))} for q in quantiles]
    if d=="lrr":
        a=np.concatenate([-feat(r,"logp") for r in clean]); b=np.concatenate([feat(r,"log_rank") for r in clean])
        return out+[{"nll_upper":float(np.quantile(a,q)),"log_rank_upper":float(np.quantile(b,z))} for q in quantiles for z in quantiles]
    a=np.concatenate([direction*local(r,d) for r in clean])
    return out+[{"lower":float(np.quantile(a,1-q))} for q in quantiles]

def _rows(rows,s,h=None,c=None):
    z=[r for r in rows if split(r)==s]
    if h is not None:z=[r for r in z if human(r)==h]
    if c is not None:z=[r for r in z if condition(r)==c]
    return z

def _thresholds(cal,d,direction,spec,clipping_mode="universal",bin_index=None,bin_label=None):
    by=defaultdict(list)
    for r in cal:by[domain(r)].append(r)
    out={}; summary=[]
    for dom,rs in sorted(by.items()):
        a=[oriented_document_score(r,d,direction) for r in rs]
        b=[oriented_document_score(r,d,direction,spec) for r in rs]
        ta=calibration_threshold(a,TARGET_FPR);tb=calibration_threshold(b,TARGET_FPR)
        out[dom]={"raw":ta,"clipped":tb}
        summary.append({"detector":d,"domain":dom,"clipping_mode":clipping_mode,
          "contamination_bin_index":bin_index,"contamination_bin":bin_label,
          "target_fpr":TARGET_FPR,"n_calibration_human":len(rs),
          "raw_threshold":ta,"raw_calibration_fpr":actual_fpr(a,ta),
          "clipped_threshold":tb,"clipped_calibration_fpr":actual_fpr(b,tb)})
    return out,summary

def _cache(rows,d,direction,spec):
    return ({id(r):oriented_document_score(r,d,direction) for r in rows},
            {id(r):oriented_document_score(r,d,direction,spec) for r in rows})

def _point(h,m,a,b,thresholds):
    ah=[a[id(r)] for r in h];bh=[b[id(r)] for r in h]
    am=[a[id(r)] for r in m];bm=[b[id(r)] for r in m]
    at=float(np.mean([a[id(r)]>=thresholds[domain(r)]["raw"] for r in m])) if m else float("nan")
    bt=float(np.mean([b[id(r)]>=thresholds[domain(r)]["clipped"] for r in m])) if m else float("nan")
    af=float(np.mean([a[id(r)]>=thresholds[domain(r)]["raw"] for r in h]))
    bf=float(np.mean([b[id(r)]>=thresholds[domain(r)]["clipped"] for r in h]))
    aa=auroc(ah,am);ba=auroc(bh,bm)
    return {"raw_tpr":at,"clipped_tpr":bt,"paired_tpr_difference":bt-at,
      "raw_auroc":aa,"clipped_auroc":ba,"paired_auroc_difference":ba-aa,
      "raw_test_fpr":af,"clipped_test_fpr":bf}

def _seed(base,*parts):
    x=int(base)&0xffffffff
    for p in parts:
        for q in str(p).encode():x=((x*16777619)^q)&0xffffffff
    return x

def _bootstrap(h,m,a,b,thresholds,reps,seed):
    if reps<=0:return {}
    ids=sorted(source(r) for r in h); pos={x:i for i,x in enumerate(ids)}
    hi=np.arange(len(ids)); domain_indices=defaultdict(list)
    hm={source(r):r for r in h}
    for sid in ids:domain_indices[domain(hm[sid])].append(pos[sid])
    ms=np.asarray([pos[source(r)] for r in m],dtype=np.int64)
    ah=np.asarray([a[id(hm[x])] for x in ids]);bh=np.asarray([b[id(hm[x])] for x in ids])
    am=np.asarray([a[id(r)] for r in m]);bm=np.asarray([b[id(r)] for r in m])
    ahit=np.asarray([ah[i]>=thresholds[domain(hm[x])]["raw"] for i,x in enumerate(ids)])
    bhit=np.asarray([bh[i]>=thresholds[domain(hm[x])]["clipped"] for i,x in enumerate(ids)])
    amit=np.asarray([am[i]>=thresholds[domain(r)]["raw"] for i,r in enumerate(m)])
    bmit=np.asarray([bm[i]>=thresholds[domain(r)]["clipped"] for i,r in enumerate(m)])
    def auc_plan(hs,mscores):
        order=np.argsort(hs,kind="mergesort"); sh=hs[order]
        return order,np.searchsorted(sh,mscores,"left"),np.searchsorted(sh,mscores,"right")
    plans=(auc_plan(ah,am),auc_plan(bh,bm))
    def weighted_auc(hw,mw,plan):
        order,left,right=plan; cum=np.r_[0,np.cumsum(hw[order])]
        favorable=cum[left]+.5*(cum[right]-cum[left])
        denom=hw.sum()*mw.sum()
        return float(np.dot(mw,favorable)/denom) if denom else float("nan")
    rng=np.random.default_rng(seed); vals=defaultdict(list)
    for _ in range(reps):
        w=np.zeros(len(ids),dtype=np.int64)
        for dom in sorted(domain_indices):
            ix=np.asarray(domain_indices[dom]);w[ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)))
        mw=w[ms]; hden=w.sum();mden=mw.sum()
        af=float(np.dot(w,ahit)/hden);bf=float(np.dot(w,bhit)/hden)
        at=float(np.dot(mw,amit)/mden) if mden else float("nan")
        bt=float(np.dot(mw,bmit)/mden) if mden else float("nan")
        aa=weighted_auc(w,mw,plans[0]);ba=weighted_auc(w,mw,plans[1])
        p={"raw_tpr":at,"clipped_tpr":bt,"paired_tpr_difference":bt-at,
          "raw_auroc":aa,"clipped_auroc":ba,"paired_auroc_difference":ba-aa,
          "raw_test_fpr":af,"clipped_test_fpr":bf}
        for k,v in p.items(): vals[k].append(v)
    return {k:percentile_interval(v) for k,v in vals.items()}

def _add_ci(row,ci):
    for k,(lo,hi) in ci.items():row[k+"_ci_low"]=lo;row[k+"_ci_high"]=hi

def _validate(rows,expected=None):
    if not rows:raise ValueError("no RAID score rows")
    ss=defaultdict(set)
    for r in rows:ss[source(r)].add(split(r))
    if any(len(v)!=1 for v in ss.values()):raise ValueError("source leakage across splits")
    if set().union(*ss.values())!={"clipping_tuning","calibration","test"}:raise ValueError("invalid split set")
    human_domains={s:{domain(r) for r in rows if split(r)==s and human(r)}
                   for s in ("clipping_tuning","calibration","test")}
    if len({tuple(sorted(v)) for v in human_domains.values()})!=1:
        raise ValueError(f"human domains differ across splits: {human_domains}")
    attacks=sorted({condition(r) for r in rows if not human(r)}-{"none"})
    normalized_expected=(
        {condition({"label":"llm","attack":x}) for x in expected}
        if expected is not None else set(attacks)
    )
    if set(attacks)!=normalized_expected:raise ValueError("configured attacks disagree with data")
    if len(attacks)!=11:raise ValueError("RAID protocol requires exactly 11 attacks")
    # Every source must have exactly one human and one machine row per condition.
    grouped=defaultdict(list)
    for r in rows:grouped[source(r)].append(r)
    for sid,sr in grouped.items():
        if sum(human(r) for r in sr)!=1 or Counter(condition(r) for r in sr if not human(r))!=Counter(["none"]+attacks):
            raise ValueError(f"incomplete source family {sid}")
    return attacks,{"score_rows":len(rows),"sources":len(ss),"human_rows":sum(human(r) for r in rows),
      "machine_rows":sum(not human(r) for r in rows),"attacks":attacks,
      "splits":dict(Counter(split(r) for r in rows)),"target_fprs":[TARGET_FPR],"detectors":list(DETECTORS)}

def _csv(path,rows):
    temporary=path.with_suffix(path.suffix+".tmp")
    if not rows:
        temporary.write_text("",encoding="utf-8");os.replace(temporary,path);return
    fields=list(rows[0]);fields+=sorted(set().union(*(set(r) for r in rows))-set(fields))
    with temporary.open("w",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    os.replace(temporary,path)
