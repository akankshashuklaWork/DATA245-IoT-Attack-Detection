"""Regenerate all data charts used in the DATA245 midterm presentation.

Runs WITHOUT refitting the ML models. Inputs are saved, audited experiment data:
  results/class_distribution_original.csv
  results/baseline_metrics.csv
  results/data_audit.json
  diagnostics/class_pattern_diversity.csv
  diagnostics/original_stage3_mitm_prediction_breakdown.csv
  diagnostics/repeated_per_class_scores.csv

Usage: python src/make_presentation_charts.py

These PNGs are reproducible visualization equivalents of the editable-shape
charts in the PowerPoint. They may have a different graphical style, but all
numeric labels and data values are computed from the saved results.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results'
DIAG=ROOT/'diagnostics'
OUT=ROOT/'presentation_charts'
OUT.mkdir(parents=True,exist_ok=True)
NAVY='#152d46'; BLUE='#3b74ac'; TEAL='#168e91'; ORANGE='#d78437'; GRAY='#8e9cac'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})

def save(fig,filename):
    fig.savefig(OUT/filename,dpi=170,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def horizontal(ax,names,values,colors,xlim=None,fmt='{:.0f}',xlabel=None):
    y=np.arange(len(names))
    bars=ax.barh(y,values,color=colors,height=.64)
    ax.set_yticks(y,names)
    ax.invert_yaxis()
    if xlim: ax.set_xlim(0,xlim)
    else: ax.set_xlim(0,max(values)*1.21)
    if xlabel: ax.set_xlabel(xlabel)
    ax.xaxis.grid(True,alpha=.16)
    ax.set_axisbelow(True)
    offset=max(values)*.018
    for b,v in zip(bars,values):
        ax.text(b.get_width()+offset,b.get_y()+b.get_height()/2,fmt.format(v),va='center',ha='left',fontsize=10,fontweight='bold',color=NAVY)

# Slide 7: class distribution (original CSV, prior to deduplication).
cls=pd.read_csv(RES/'class_distribution_original.csv')
cls['type']=cls['type'].astype(str)
cls=cls.sort_values('count',ascending=False,kind='stable')
assert int(cls['count'].sum())==211043, 'Unexpected class count total'
assert int(cls.loc[cls.type=='mitm','count'].iloc[0])==1043
fig,ax=plt.subplots(figsize=(11,6))
horizontal(ax,cls.type.str.upper().tolist(),cls['count'].to_numpy(),[ORANGE if k=='mitm' else (BLUE if k=='normal' else GRAY) for k in cls.type],xlim=58000,fmt='{:.0f}',xlabel='Network records, original dataset')
ax.set_title('Slide 7 | Class distribution (N=211,043)',loc='left',color=NAVY,fontweight='bold',pad=15)
save(fig,'slide07_class_distribution.png')
cls[['type','count','percentage']].to_csv(OUT/'slide07_data.csv',index=False)

# Slide 8: unique 5-feature vectors per class, AFTER full-row deduplication.
pat=pd.read_csv(DIAG/'class_pattern_diversity.csv').set_index('class')
classes=['mitm','ransomware','xss']
sub=pat.loc[classes].copy()
assert int(sub.loc['mitm','unique_five_feature_patterns'])==1007
sub['percentage']=sub['unique_five_feature_patterns']/sub['records']*100
fig,ax=plt.subplots(figsize=(10.5,4.9))
horizontal(ax,['MITM','Ransomware','XSS'],sub['percentage'].to_numpy(),[TEAL,ORANGE,BLUE],xlim=111,fmt='{:.1f}%',xlabel='Unique exact 5-feature vectors / records (%)')
ax.set_title('Slide 8 | Feature-pattern diversity, deduplicated data',loc='left',color=NAVY,fontweight='bold',pad=15)
fig.subplots_adjust(bottom=.27)
fig.text(.12,.015,'Note: distinct numeric combinations do not imply independent attack sessions.',size=9,color='#5a6570')
save(fig,'slide08_pattern_diversity.png')
sub[['records','unique_five_feature_patterns','percentage']].reset_index().to_csv(OUT/'slide08_data.csv',index=False)

# Slide 10: baseline DT and kNN scores from untampered test split.
base=pd.read_csv(RES/'baseline_metrics.csv').query('split=="test"').set_index('model')
rows=[('Accuracy','accuracy'),('Macro-F1','macro_f1'),('MITM recall','mitm_recall')]
vals=np.array([[base.loc['Decision_Tree',col]*100,base.loc['kNN',col]*100] for _,col in rows])
fig,ax=plt.subplots(figsize=(10.5,5))
y=np.arange(len(rows)); ax.barh(y-.18,vals[:,0],.33,color=BLUE,label='Decision Tree'); ax.barh(y+.18,vals[:,1],.33,color=TEAL,label='kNN')
for i in range(len(rows)):
    for j,color in enumerate([BLUE,TEAL]): ax.text(vals[i,j]+1.0,i+(-.18 if j==0 else .18),f'{vals[i,j]:.1f}%',va='center',fontsize=10,color=color,fontweight='bold')
ax.set_yticks(y,[r[0] for r in rows]);ax.invert_yaxis();ax.set_xlim(0,109);ax.set_xlabel('Held-out test score (%)')
ax.set_title('Slide 10 | Five-feature baseline results',loc='left',color=NAVY,fontweight='bold',pad=15)
ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.legend(frameon=False,loc='lower right')
save(fig,'slide10_baseline_comparison.png')
pd.DataFrame({'metric':[r[0] for r in rows],'Decision_Tree_pct':vals[:,0],'kNN_pct':vals[:,1]}).to_csv(OUT/'slide10_data.csv',index=False)

# Slide 11 (left): misclassification of true MITM, decision tree test predictions.
mis=pd.read_csv(DIAG/'original_stage3_mitm_prediction_breakdown.csv')
mis=mis.query('model=="Decision_Tree" and split=="test"').set_index('predicted_class')['n']
assert int(mis.sum())==149 and int(mis.loc['mitm'])==104
categories={'Normal':int(mis.get('normal',0)),'Password':int(mis.get('password',0)),'DoS':int(mis.get('dos',0)),'Injection':int(mis.get('injection',0))}
categories['Other']=int(mis.sum()-mis.get('mitm',0)-sum(categories.values()))
assert sum(categories.values())==45
fig,ax=plt.subplots(figsize=(9.5,4.6))
horizontal(ax,list(categories),list(categories.values()),[ORANGE,ORANGE,ORANGE,GRAY,GRAY],xlim=15,fmt='{:.0f}',xlabel='Actual MITM cases assigned to another label')
ax.set_title('Slide 11 | How the Decision Tree missed 45 MITM cases',loc='left',color=NAVY,fontweight='bold',pad=13)
save(fig,'slide11_mitm_errors.png')
pd.DataFrame({'predicted_label':list(categories),'missed_mitm_records':list(categories.values())}).to_csv(OUT/'slide11_mitm_errors_data.csv',index=False)

# Slide 11 (right): repeated group-disjoint split diagnostics.
reps=pd.read_csv(DIAG/'repeated_per_class_scores.csv')
reps=reps[reps.model=='Decision Tree']
classes=['mitm','ransomware','xss','injection']
stats=reps.groupby('class')['recall'].agg(['mean','min','max','count']).loc[classes]*100
stats['count']=stats['count']/100
assert (stats['count']==10).all()
fig,ax=plt.subplots(figsize=(9.5,4.6))
horizontal(ax,['MITM','Ransomware','XSS','Injection'],stats['mean'].to_numpy(),[TEAL,ORANGE,BLUE,BLUE],xlim=112,fmt='{:.1f}%',xlabel='Mean Decision Tree recall over 10 group-disjoint splits (%)')
ax.set_title('Slide 11 | Repeated-split difficult-class recall',loc='left',color=NAVY,fontweight='bold',pad=13)
save(fig,'slide11_repeated_split_mean_recall.png')
stats.to_csv(OUT/'slide11_repeated_split_data.csv',float_format='%.5f')

meta={'slides':{
   '7':'results/class_distribution_original.csv',
   '8':'diagnostics/class_pattern_diversity.csv + results/data_audit.json',
   '10':'results/baseline_metrics.csv (test rows only)',
   '11 (errors)':'diagnostics/original_stage3_mitm_prediction_breakdown.csv',
   '11 (mean recall)':'diagnostics/repeated_per_class_scores.csv',
 },'note':'Figures in the PowerPoint are editable shapes, drawn separately using rounded values from the above files. These plots recreate the quantities, not the exact PowerPoint design.'}
(OUT/'chart_provenance.json').write_text(json.dumps(meta,indent=2))
print('DONE: regenerated 5 data charts and their numeric source tables in',OUT)
for k in meta['slides']: print('Slide',k,'source:',meta['slides'][k])
