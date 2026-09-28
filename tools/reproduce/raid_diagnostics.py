"""Regenerate numerical inputs for Appendix G.1, G.7 and G.8."""
import csv
from collections import Counter
import numpy as np

def read(path):
    with path.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))

def write(path,rows):
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def build_diagnostics(source,output):
    write(output/'table_14_raw_baseline.csv',read(source/'binoculars_sanity.csv'))
    records=[r for r in read(source/'contamination_records.csv') if r['split']=='test']
    summaries=[]
    for attack in sorted({r['attack'] for r in records}):
        group=[r for r in records if r['attack']==attack]
        if len(group)!=5149:raise ValueError('Wrong per-attack test count')
        rates=np.array([float(r['realized_contamination_rate']) for r in group])
        q=np.quantile(rates,[.25,.5,.75])
        summaries.append({'condition':attack,'n_test':len(group),'median':q[1],'q1':q[0],'q3':q[2],
            'zero_percent':100*np.mean(rates==0),'eligible_percent':100*np.mean((rates>0)&(rates<=.5)),
            'dense_percent':100*np.mean(rates>.5),
            'truncated_percent':100*np.mean([r['truncation_applied_attacked'].lower()=='true' for r in group]),
            'full_distance_exceeds_window_percent':100*np.mean([int(r['full_levenshtein_distance'])>int(r['levenshtein_distance']) for r in group])})
    write(output/'table_54_rates_and_truncation.csv',summaries)
    sources=read(source/'selected_sources.csv')
    for name,key in [('table_55_generators.csv','source_generator_model'),('table_56_domains.csv','domain')]:
        counts=Counter((r[key],r['split']) for r in sources)
        rows=[]
        for label in sorted({r[key] for r in sources}):
            row={key:label,**{split:counts[label,split] for split in ['clipping_tuning','calibration','test']}}
            row['total']=sum(v for k,v in row.items() if k!=key);rows.append(row)
        write(output/name,rows)
