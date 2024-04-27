import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import mlflow
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import os
import mlflow

df=pd.read_csv('data.csv',sep=';')

df=df.astype({'Marital status':str, 'Application mode':str, 'Application order':str, 'Course':str,
       'Daytime/evening attendance\t':str, 'Previous qualification':str,
       'Previous qualification (grade)':str, 'Nacionality':str,
       "Mother's qualification":str, "Father's qualification":str,
       "Mother's occupation":str, "Father's occupation":str, 'Admission grade':str,
       'Displaced':str, 'Educational special needs':str, 'Debtor':str,
       'Tuition fees up to date':str, 'Gender':str, 'Scholarship holder':str, 'International':str,
       'Curricular units 1st sem (credited)':str,
       'Curricular units 1st sem (enrolled)':str,
       'Curricular units 1st sem (evaluations)':str,
       'Curricular units 1st sem (approved)':str,
       'Curricular units 1st sem (grade)':str,
       'Curricular units 1st sem (without evaluations)':str,
       'Curricular units 2nd sem (credited)':str,
       'Curricular units 2nd sem (enrolled)':str,
       'Curricular units 2nd sem (evaluations)':str,
       'Curricular units 2nd sem (approved)':str,
       'Curricular units 2nd sem (grade)':str,
       'Curricular units 2nd sem (without evaluations)':str
       }) 

df['Target']= df['Target'].replace('Enrolled', 0)
df['Target']= df['Target'].replace('Graduate', 0)
df['Target']= df['Target'].replace('Dropout', 1)

x=df.drop('Target',axis=1)
y=df['Target']

scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,train_size=0.8,random_state=42)

#rf
rf=RandomForestClassifier(random_state=42)

param_grid_forest={
    'n_estimators':[200,400,700],
    'max_depth':[10,20,30],
    'criterion':['gini','entropy'],
    'max_leaf_nodes' :[50,100]
}

grid_forest=GridSearchCV(estimator=rf,
                         param_grid=param_grid_forest,
                         cv=5,
                         n_jobs=-1,
                         scoring='accuracy',verbose=0)

model_forest=grid_forest.fit(x_train,y_train)

#lr
lr=LogisticRegression(random_state=42)

param_grid_lr={
    'C':[100,10,1,0.1],
    'penalty':['l1','l2'],
    'solver':['liblinear']
}

grid_log=GridSearchCV(estimator=lr,
                         param_grid=param_grid_lr,
                         cv=5,
                         n_jobs=-1,
                         scoring='accuracy',verbose=0)

model_log=grid_log.fit(x_train,y_train)

#dt
dt=DecisionTreeClassifier(random_state=42)
param_grid_tree={
    'max_depth':[3,5,7,9,11],
    'criterion':['gini','entropy']
}

grid_tree=GridSearchCV(estimator=dt,
                         param_grid=param_grid_tree,
                         cv=5,
                         n_jobs=-1,
                         scoring='accuracy',verbose=0)

model_tree=grid_tree.fit(x_train,y_train)


#mlflow

mlflow.set_experiment('dropout-predict')

#model evaluation metric
def eval_metrics(actual, pred):
    accuracy = metrics.accuracy_score(actual, pred)
    f1 = metrics.f1_score(actual, pred, pos_label=1)
    fpr, tpr, _ = metrics.roc_curve(actual, pred)
    auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, color='blue', label='ROC curve area = %0.2f'%auc)
    plt.plot([0,1],[0,1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend(loc='lower right')
    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/ROC_curve.png")
    # Close plot
    plt.close()
    return(accuracy, f1, auc)

def mlflow_logging(model,x,y,name):
    with mlflow.start_run() as run :
        run_id=run.info.run_id
        mlflow.set_tag('run id',run_id)
        pred=model.predict(x)

        accuracy,f1,auc=eval_metrics(y,pred)
        mlflow.log_params(model.best_params_)

        mlflow.log_metric('mean cv score',model.best_score_)
        mlflow.log_metric('accuracy',accuracy)
        mlflow.log_metric('f1',f1)
        mlflow.log_metric('auc',auc)

        mlflow.log_artifact('plots/ROC_CURVE.png')
        mlflow.sklearn.log_model(model,name)

        mlflow.end_run()

mlflow_logging(model_tree,x_test,y_test,'decision tree')
mlflow_logging(model_log,x_test,y_test,'logreg')
mlflow_logging(model_forest,x_test,y_test,'rf classifier')

