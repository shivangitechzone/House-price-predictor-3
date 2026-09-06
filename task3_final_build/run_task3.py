from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

BASE=Path(__file__).resolve().parent
DATA=BASE/'data'; RESULTS=BASE/'results'; FIG=BASE/'figures'; MODELS=BASE/'models'
for p in (RESULTS,FIG,MODELS): p.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(DATA/'california_housing.csv')
X=df.drop('MedHouseVal',axis=1); y=df['MedHouseVal']
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
rmse=lambda a,b: float(np.sqrt(mean_squared_error(a,b)))

def pipe(model): return Pipeline([('scaler',StandardScaler()),('model',model)])
lr=pipe(LinearRegression()).fit(Xtr,ytr); ridge=pipe(Ridge(alpha=1.0)).fit(Xtr,ytr)
lr_p=lr.predict(Xte); ridge_p=ridge.predict(Xte)

default=DecisionTreeRegressor(random_state=42).fit(Xtr,ytr)
default_tr=rmse(ytr,default.predict(Xtr)); default_te=rmse(yte,default.predict(Xte))
default_tr_r2=r2_score(ytr,default.predict(Xtr)); default_te_r2=r2_score(yte,default.predict(Xte))
cv=-cross_val_score(default,Xtr,ytr,cv=5,scoring='neg_root_mean_squared_error',n_jobs=-1)
pipeline=pipe(DecisionTreeRegressor(random_state=42))
params={'model__max_depth':[3,5,7,10],'model__min_samples_split':[2,5,10]}
grid=GridSearchCV(pipeline,params,scoring='neg_root_mean_squared_error',cv=5,n_jobs=-1,return_train_score=True).fit(Xtr,ytr)
best=grid.best_estimator_; bp=best.predict(Xte)
ttr=rmse(ytr,best.predict(Xtr)); tte=rmse(yte,bp); ttr2=r2_score(ytr,best.predict(Xtr)); tte2=r2_score(yte,bp)
comp=pd.DataFrame({'Model':['Linear Regression','Ridge Regression','Tuned Decision Tree'],'RMSE':[rmse(yte,lr_p),rmse(yte,ridge_p),tte],'R2 Score':[r2_score(yte,lr_p),r2_score(yte,ridge_p),tte2],'Overfit Gap':[np.nan,np.nan,tte-ttr]})
comp.to_csv(RESULTS/'model_comparison.csv',index=False)
pd.DataFrame({'Fold':range(1,6),'RMSE':cv}).to_csv(RESULTS/'cross_validation_scores.csv',index=False)
gr=pd.DataFrame(grid.cv_results_); gr['mean_test_RMSE']=-gr.mean_test_score; gr['std_test_RMSE']=gr.std_test_score; gr['mean_train_RMSE']=-gr.mean_train_score; gr.sort_values('mean_test_RMSE').to_csv(RESULTS/'grid_search_results.csv',index=False)
pd.DataFrame([{'Parameter':k,'Best Value':v} for k,v in grid.best_params_.items()]).to_csv(RESULTS/'best_parameters.csv',index=False)
pd.DataFrame({'Metric':['Default Tree Train RMSE','Default Tree Test RMSE','Default Tree Train R2','Default Tree Test R2','Default Tree 5-Fold CV Mean RMSE','Default Tree 5-Fold CV Std RMSE','Tuned Tree Train RMSE','Tuned Tree Test RMSE','Tuned Tree Train R2','Tuned Tree Test R2','Tuned Tree 5-Fold CV Mean RMSE','Tuned Tree Test Overfit Gap'],'Value':[default_tr,default_te,default_tr_r2,default_te_r2,cv.mean(),cv.std(),ttr,tte,ttr2,tte2,-grid.best_score_,tte-ttr]}).to_csv(RESULTS/'task3_summary.csv',index=False)
joblib.dump(best,MODELS/'tuned_decision_tree.joblib')
plt.figure(figsize=(8,5)); plt.bar(['Train RMSE','Test RMSE'],[default_tr,default_te]); plt.title('Default Decision Tree: Train vs Test RMSE'); plt.ylabel('RMSE'); plt.tight_layout(); plt.savefig(FIG/'overfitting_train_vs_test.png',dpi=180); plt.close()
plt.figure(figsize=(9,5)); plt.bar(comp.Model,comp.RMSE); plt.title('Final Model Comparison — RMSE'); plt.ylabel('RMSE'); plt.xticks(rotation=15); plt.tight_layout(); plt.savefig(FIG/'model_comparison_rmse.png',dpi=180); plt.close()
plt.figure(figsize=(7,6)); plt.scatter(yte,bp,alpha=.35,s=12); mn=min(yte.min(),bp.min()); mx=max(yte.max(),bp.max()); plt.plot([mn,mx],[mn,mx],linestyle='--'); plt.xlabel('Actual House Value'); plt.ylabel('Predicted House Value'); plt.title('Tuned Decision Tree: Actual vs Predicted'); plt.tight_layout(); plt.savefig(FIG/'actual_vs_predicted_tuned_tree.png',dpi=180); plt.close()
print('BEST',grid.best_params_); print('CV',-grid.best_score_); print(comp.round(4).to_string(index=False))
