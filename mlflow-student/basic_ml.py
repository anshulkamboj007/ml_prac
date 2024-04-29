import io
import mlflow
import argparse
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

"""
def load_data():
    URL="http://archive.ics.uci.edu/ml/machine-learning-database/wine-quality/winequality-red.csv"
    try:
        df=pd.read_csv('winequality.csv',sep=';')
    except Exception as e:
        raise e
"""

def main(alpha,l1_ratio):
    df=pd.read_csv('winequality.csv',sep=';')
    TARGET="quality"
    X=df.drop(columns=TARGET)
    y=df[TARGET]
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=6)
    

    mlflow.set_experiment('ml-model-elastnet-1')
    with mlflow.start_run():
        mlflow.log_param('alpha',alpha)
        mlflow.log_param('l1_ratio',l1_ratio)

        model =ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=6)
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
        rmse,mae,r2=eval_function(y_test,y_pred)

        mlflow.log_metric('rmse',rmse)
        mlflow.log_metric('mae',mae)
        mlflow.log_metric('r2 score',r2)

        mlflow.sklearn.log_model(model,'trained_model')
        print('-----------completed---------------')


def eval_function(actual,pred):
    rmse= mean_squared_error(actual,pred,squared=False)
    mae=mean_absolute_error(actual,pred)
    r2=r2_score(actual,pred)

    return rmse,mae,r2



if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument("--alpha",'-a',type=float,default=0.2)
    args.add_argument('--l1_ratio','-p2',type=float,default=0.3)
    parsed_args=args.parse_args()

    main(parsed_args.alpha,parsed_args.l1_ratio)