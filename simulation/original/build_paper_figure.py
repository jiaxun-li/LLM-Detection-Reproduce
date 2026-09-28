"""Build Appendix C's vector figure source from frozen simulation CSV."""
import csv
from pathlib import Path

root=Path(__file__).resolve().parent
rows=list(csv.DictReader((root/'results/monte_carlo.csv').open()))
titles={'loglik':'Log-likelihood','negative_rank':'Rank','negative_logrank':'Log-rank','negative_entropy_gap':'Entropy gap'}
parts=[r'\documentclass[tikz,border=3pt]{standalone}',r'\usepackage{pgfplots}',
       r'\usepgfplotslibrary{groupplots}',r'\pgfplotsset{compat=1.18}',
       r'\begin{document}\begin{tikzpicture}',
       r'\begin{groupplot}[group style={group size=4 by 2,horizontal sep=0.8cm,vertical sep=2cm},width=4cm,height=4cm,xmode=log,log basis x=2,xtick={128,512,2048,4096},xticklabels={128,512,2k,4k},ymin=0,ymax=1.03,ytick={0,.5,1},grid=major,tick label style={font=\scriptsize},title style={font=\small,align=center},label style={font=\small}]']
for K in (10,100):
    for detector,title in titles.items():
        ylabel='Power' if detector=='loglik' else ''
        parts.append(r'\nextgroupplot[title={'+title+r'\\$K='+str(K)+r'$},xlabel={Length $n$},ylabel={'+ylabel+'}]')
        for agg,style in [('raw','blue,solid,mark=o'),('clipped','orange!85!black,dashed,mark=square*')]:
            selected=sorted((r for r in rows if r['profile']=='mild' and r['support_size']=='1000'
                             and r['K']==str(K) and r['detector']==detector and r['theorem_selected']=='True'
                             and r['attack']=='minimum_force_zero' and r['calibration']=='calibrated_5pct'
                             and r['aggregation']==agg),key=lambda r:int(r['n']))
            assert len(selected)==4
            coords=' '.join(f"({r['n']},{r['tpr']})" for r in selected)
            parts.append(r'\addplot['+style+r',thick] coordinates {'+coords+'};')
parts.extend([r'\end{groupplot}\end{tikzpicture}\end{document}'])
dest=root/'figure_build/section4_alt_power.tex'
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text('\n'.join(parts),encoding='utf-8')
print(dest)
