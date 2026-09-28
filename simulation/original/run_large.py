"""Large-vocabulary, genuine order-K validation of the Section 4 theorem."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from analysis import audit, floor_for, analytic_attack_mean
from models import make_profiles, score_arrays
from simulate import simulate


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,keys); w.writeheader(); w.writerows(rows)


def wilson(k,n):
    z=1.959963984540054; p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    width=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,center-width),min(1.,center+width)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--paths',type=int,default=2000)
    ap.add_argument('--calibration',type=int,default=4000)
    ap.add_argument('--lengths',type=int,nargs='+',default=[128,512,2048,4096])
    ap.add_argument('--orders',type=int,nargs='+',default=[10,100])
    ap.add_argument('--supports',type=int,nargs='+',default=[500,1000])
    ap.add_argument('--seed',type=int,default=20260906)
    ap.add_argument('--output',default='results')
    args=ap.parse_args()
    root=Path(__file__).resolve().parent
    out=(root/args.output).resolve()
    if not out.is_relative_to(root):
        raise ValueError('Output must remain within simulation_section4_large')
    out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    nextseed=lambda:int(rng.integers(0,2**32-1))
    audits=[]; exact=[]; rows=[]; coverage=[]; profiles=[]
    start=time.time(); checks=0
    config=dict(vars(args),vocabulary_size=10000,initial_history='K zero tokens',
                numpy_version=np.__version__,
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')})
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    for M in args.supports:
        for profile in make_profiles(M):
            p,q=profile['p'],profile['q']
            profiles.append(dict(name=profile['name'],support_size=M,p=p.tolist(),q=q.tolist(),
                                 description=profile['description']))
            mapping=score_arrays(p); names=list(mapping); scores=np.stack(list(mapping.values()))
            cert=[audit(p,q,s) for s in scores]
            # Select a common contamination level when all detector intervals overlap.
            lo=max(a['epsilon_raw'] for a in cert)
            hi=min(min(1.,a['epsilon_clip']) for a in cert)
            common=all(a['certified'] for a in cert) and hi-lo>1e-9
            centers=[(a['epsilon_raw']+min(1.,a['epsilon_clip']))/2 if a['certified'] else None for a in cert]
            floors=np.array([floor_for(a,e) if e is not None else a['b']+a['r']/2 for a,e in zip(cert,centers)])
            epsilons=sorted(set(e for e in centers if e is not None)) or [.2,.4,.6]
            for K in args.orders:
                tag=dict(profile=profile['name'],support_size=M,K=K,vocabulary_size=10000)
                print(f"M={M} K={K} {profile['name']} certified={sum(a['certified'] for a in cert)}/4",flush=True)
                for name,a,floor,center in zip(names,cert,floors,centers):
                    audits.append(dict(**tag,detector=name,**a,common_interval=common,
                                       selected_epsilon=center,clipping_floor=float(floor)))
                cal,meta=simulate(p,q,scores,floors,K,args.lengths,args.calibration,nextseed(),source='human')
                coverage.append({**tag,'role':'calibration','epsilon':0,**meta})
                human,meta=simulate(p,q,scores,floors,K,args.lengths,args.paths,nextseed(),source='human')
                coverage.append({**tag,'role':'human_test','epsilon':0,**meta})
                for epsilon in epsilons:
                    for attack in ['minimum_zero','minimum_force_zero','human_mixture']:
                        draws,meta=simulate(p,q,scores,floors,K,args.lengths,args.paths,nextseed(),
                                            source='machine',epsilon=epsilon,attack=attack)
                        coverage.append({**tag,'role':'machine_test','epsilon':epsilon,**meta})
                        for j,(name,score,a,floor) in enumerate(zip(names,scores,cert,floors)):
                            selected=bool(centers[j] is not None and abs(epsilon-centers[j])<1e-12)
                            clipped=np.maximum(score,floor); t=floor-a['b']
                            human_mean=a['mu0']; human_clip=float((q*clipped).sum()/2)
                            raw_mean=analytic_attack_mean(p,q,score,epsilon,K,attack)
                            clip_mean=analytic_attack_mean(p,q,clipped,epsilon,K,attack)
                            certified_gap=(1-epsilon)*a['mean_min']+epsilon*floor-human_mean-a['rho']*t
                            assert clip_mean-human_clip>=certified_gap-1e-9
                            checks+=1
                            if selected:
                                assert certified_gap>0
                                if attack!='human_mixture':
                                    assert raw_mean<human_mean
                                checks+=1
                            exact.append(dict(**tag,detector=name,epsilon=epsilon,attack=attack,
                                theorem_selected=selected,raw_mean=raw_mean,clipped_mean=clip_mean,
                                human_mean=human_mean,human_clipped_mean=human_clip,
                                raw_gap=raw_mean-human_mean,clipped_gap=clip_mean-human_clip,
                                uniform_adaptive_clipped_lower_gap=certified_gap))
                            for n in args.lengths:
                                thresholds={}
                                for agg in ['raw','clipped']:
                                    thresholds[agg]=float(np.quantile(cal[n][agg][:,j],.95,method='higher'))
                                threshold_sets={'calibrated_5pct':thresholds}
                                if selected:
                                    hbound=human_mean+a['rho']*t
                                    abound=(1-epsilon)*a['mean_min']+epsilon*floor
                                    threshold_sets['fixed_theorem']={'raw':human_mean+.01*np.ptp(score),
                                                                     'clipped':(hbound+abound)/2}
                                for calibration,thresholds in threshold_sets.items():
                                    for agg in ['raw','clipped']:
                                        threshold=thresholds[agg]
                                        successes=int((draws[n][agg][:,j]>threshold).sum())
                                        false=int((human[n][agg][:,j]>threshold).sum())
                                        tl,tu=wilson(successes,args.paths); fl,fu=wilson(false,args.paths)
                                        rows.append(dict(**tag,detector=name,epsilon=epsilon,attack=attack,
                                            theorem_selected=selected,n=n,aggregation=agg,calibration=calibration,
                                            threshold=threshold,tpr=successes/args.paths,tpr_low=tl,tpr_high=tu,
                                            fpr=false/args.paths,fpr_low=fl,fpr_high=fu,
                                            paths=args.paths,calibration_paths=args.calibration,
                                            sample_mean=float(draws[n][agg][:,j].mean())))
                write_csv(out/'assumption_audit.csv',audits)
                write_csv(out/'exact_means.csv',exact)
                write_csv(out/'monte_carlo.csv',rows)
                (out/'coverage.json').write_text(json.dumps(coverage,indent=2),encoding='utf-8')
    (out/'profiles.json').write_text(json.dumps(profiles,indent=2),encoding='utf-8')
    done=dict(status='complete',elapsed_seconds=time.time()-start,assumption_rows=len(audits),
              exact_rows=len(exact),monte_carlo_rows=len(rows),checks=checks,
              certified=sum(a['certified'] for a in audits),strong=sum(a['strong_condition'] for a in audits),
              vocabulary_coverage_min=min(c['visited_token_count'] for c in coverage),
              vocabulary_coverage_max=max(c['visited_token_count'] for c in coverage),
              parity_checks=all(c['final_history_parity_check'] for c in coverage))
    (out/'completion.json').write_text(json.dumps(done,indent=2),encoding='utf-8')
    print(json.dumps(done),flush=True)


if __name__=='__main__':
    main()
