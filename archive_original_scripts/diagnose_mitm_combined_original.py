"""Reproducible MITM diversity and ten repeated leakage-aware evaluations.
Only uses five numerical input features. Based on run_project.py baseline models.
"""
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

ROOT=Path(__file__).resolve().parent
DATA_CSV=Path(os.environ.get('TONIOT_CSV',ROOT.parent/'data/train_test_network.csv'))
STAGE3_PRED_CSV=Path(os.environ.get('TONIOT_STAGE3_PREDS',ROOT.parent/'results/predictions_val_test.csv'))
FEATURES=['duration','src_bytes','dst_bytes','src_ip_bytes','dst_ip_bytes']
SEEDS=[42,7,17,23,29,41,73,101,202,313]
raw=pd.read_csv(DATA_CSV)
df=raw.drop_duplicates().reset_index(drop=True)
X=df[FEATURES].apply(pd.to_numeric,errors='coerce'); y=df['type']
groups=pd.util.hash_pandas_object(df[FEATURES],index=False).to_numpy(dtype=np.uint64)
mitm=df.loc[y.eq('mitm')]; mitm_patterns=mitm[FEATURES].drop_duplicates()
pattern_stats=df.groupby(FEATURES,dropna=False,sort=False)['type'].agg(nrows='size',nlabels='nunique').reset_index()
conflict_pattern_set=pattern_stats.loc[pattern_stats['nlabels']>1,FEATURES]
mitm_conflict=mitm_patterns.merge(conflict_pattern_set,on=FEATURES,how='inner')
mitm_conflict_rows=mitm.merge(mitm_conflict,on=FEATURES,how='inner')
matching=df.merge(mitm_patterns,on=FEATURES,how='inner')
matching_conflicting=df.merge(mitm_conflict,on=FEATURES,how='inner')
class_patterns=[]
for cls,sub in df.groupby('type'):
    uniq=sub[FEATURES].drop_duplicates()
    conf=uniq.merge(conflict_pattern_set,on=FEATURES,how='inner')
    class_patterns.append({'class':cls,'records':len(sub),'unique_five_feature_patterns':len(uniq),'pattern_diversity_ratio':len(uniq)/len(sub),'patterns_shared_with_other_labels':len(conf)})
pd.DataFrame(class_patterns).sort_values('records').to_csv(ROOT/'class_pattern_diversity.csv',index=False)
saved=pd.read_csv(STAGE3_PRED_CSV)
mis=[]
for (model, split), sub in saved.loc[saved.actual.eq('mitm')].groupby(['model','split']):
    for label,n in sub.predicted.value_counts().items():
        mis.append({'model':model,'split':split,'true_class':'mitm','predicted_class':label,'n':int(n)})
pd.DataFrame(mis).to_csv(ROOT/'original_stage3_mitm_prediction_breakdown.csv',index=False)

def get_model(model):
    if model=='Decision Tree':
        return Pipeline([('imputer',SimpleImputer(strategy='median')),('classifier',DecisionTreeClassifier(criterion='gini',random_state=42))])
    return Pipeline([('imputer',SimpleImputer(strategy='median')),('log1p',FunctionTransformer(np.log1p,validate=False)),('standardize',StandardScaler()),('classifier',KNeighborsClassifier(n_neighbors=5,algorithm='kd_tree',weights='uniform',n_jobs=1))])

# Fast group-disjoint repeated split. Each distinct 5-feature vector is one group.
# Stratify at the group level by its majority label; group sizes can differ.
majority=(pd.DataFrame({'group':groups,'label':y.to_numpy()})
          .groupby(['group','label']).size().reset_index(name='count')
          .sort_values(['group','count','label'],ascending=[True,False,True])
          .drop_duplicates('group'))
all_group_ids=majority['group'].to_numpy()
all_majority_labels=majority['label'].to_numpy()
rows=[]; class_rows=[]
for i,seed in enumerate(SEEDS,1):
    gtrain,gremain=train_test_split(all_group_ids,test_size=.30,random_state=seed,stratify=all_majority_labels)
    remaining_majority=majority.set_index('group').loc[gremain,'label'].to_numpy()
    gval,gtest=train_test_split(gremain,test_size=.5,random_state=seed+5000,stratify=remaining_majority)
    train=np.flatnonzero(np.isin(groups,gtrain)); val=np.flatnonzero(np.isin(groups,gval)); test=np.flatnonzero(np.isin(groups,gtest))
    assert len(train)+len(val)+len(test)==len(df)
    assert len(set(groups[train]) & set(groups[test]))==0
    assert len(set(groups[val]) & set(groups[test]))==0
    assert len(set(groups[train]) & set(groups[val]))==0
    labels=sorted(y.unique())
    for name in ['Decision Tree','kNN']:
        model=get_model(name)
        model.fit(X.iloc[train],y.iloc[train]); predicted=model.predict(X.iloc[test]); actual=y.iloc[test].to_numpy()
        p,r,f,s=precision_recall_fscore_support(actual,predicted,labels=labels,zero_division=0)
        mpos=labels.index('mitm')
        rows.append({'seed':seed,'model':name,'train_rows':len(train),'validation_rows':len(val),'test_rows':len(test),'mitm_train':int((y.iloc[train]=='mitm').sum()),'mitm_test':int((actual=='mitm').sum()),'mitm_correct':int(((actual=='mitm')&(predicted=='mitm')).sum()),'mitm_precision':float(p[mpos]),'mitm_recall':float(r[mpos]),'mitm_f1':float(f[mpos]),'accuracy':float(accuracy_score(actual,predicted)),'macro_f1':float(f1_score(actual,predicted,average='macro',zero_division=0))})
        for cls,precision,recall,f1,support in zip(labels,p,r,f,s):
            class_rows.append({'seed':seed,'model':name,'class':cls,'support':int(support),'precision':float(precision),'recall':float(recall),'f1':float(f1)})
    print(f'Done {i}/10: seed {seed}, test MITM {int((actual=="mitm").sum())}, test total {len(test)}',flush=True)

m=pd.DataFrame(rows); m.to_csv(ROOT/'repeated_group_safe_splits.csv',index=False)
pd.DataFrame(class_rows).to_csv(ROOT/'repeated_per_class_scores.csv',index=False)
summary=m.groupby('model').agg(runs=('seed','count'),mitm_test_min=('mitm_test','min'),mitm_test_max=('mitm_test','max'),mitm_recall_mean=('mitm_recall','mean'),mitm_recall_std=('mitm_recall','std'),mitm_recall_min=('mitm_recall','min'),mitm_recall_max=('mitm_recall','max'),mitm_precision_mean=('mitm_precision','mean'),macro_f1_mean=('macro_f1','mean'),macro_f1_std=('macro_f1','std')).reset_index()
summary.to_csv(ROOT/'repeat_summary.csv',index=False)
report={'original_rows':len(raw),'exact_duplicates':int(raw.duplicated().sum()),'deduped_rows':len(df),'mitm_original':int(raw.type.eq('mitm').sum()),'mitm_deduped':len(mitm),'mitm_unique_patterns':len(mitm_patterns),'mitm_pattern_diversity_pct':round(100*len(mitm_patterns)/len(mitm),3),'mitm_patterns_shared_with_other_labels':len(mitm_conflict),'mitm_rows_shared_with_other_labels':len(mitm_conflict_rows),'all_rows_matching_mitm_pattern_by_label':matching.type.value_counts().to_dict(),'all_rows_matching_shared_mitm_pattern_by_label':matching_conflicting.type.value_counts().to_dict(),'seeds':SEEDS,'repeat_summary':summary.to_dict('records')}
(ROOT/'summary.json').write_text(json.dumps(report,indent=2))
fig,axes=plt.subplots(1,2,figsize=(11,4.2))
for name,sub in m.groupby('model'):
    axes[0].plot(sub.seed.astype(str),sub.mitm_recall,marker='o',label=name)
    axes[1].plot(sub.seed.astype(str),sub.macro_f1,marker='o',label=name)
axes[0].set_title('MITM recall: 10 leakage-aware splits'); axes[1].set_title('Macro-F1: 10 leakage-aware splits')
for ax in axes:
    ax.set_ylim(0,1);ax.set_ylabel('Score');ax.set_xlabel('Seed');ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(ROOT/'repeated_split_stability.png',dpi=155);plt.close(fig)
print('SUMMARY',json.dumps(report,indent=2),flush=True)
