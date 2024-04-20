from numpy import ndarray
from sklearn.base import BaseEstimator,TransformerMixin
from prediction_model.config import config
import numpy as np


class MeanImputer(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables=variables

    def fit(self,X,y=None):
        self.mean_dict ={}
        for col in self.variables :
            self.mean_dict[col] =X[col].mean()
        return self
    
    def transform(self,X):
        X=X.copy()
        for col in self.variables:
            X[col].fillna(self.mean_dict[col],inplace=True)
        return X
    
class ModeImputer(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables=variables

    def fit(self,X,y=None):
        self.mode_dict ={}
        for col in self.variables :
            self.mode_dict[col] =X[col].mode()
        return self
    
    def transform(self,X):
        X=X.copy()
        for col in self.variables:
            X[col].fillna(self.mode_dict[col],inplace=True)
        return X
    
class DropColumns(BaseEstimator,TransformerMixin):
    def __init__(self,variables_to_drop=None):
        self.variables_to_drop=variables_to_drop

    def fit(self,X,y=None):

        return self
    
    def transform(self,X):
        X=X.copy()
        X=X.drop(columns = self.variables_to_drop)

        return X
    
class ChangetoSTR(BaseEstimator,TransformerMixin):
    def __init__(self,variables_to_change=None):
        self.variables_to_change=variables_to_change

    def fit(self,X,y=None):

        return self
    
    def transform(self,X):
        X=X.copy()
        for col in self.variables_to_change:
            X[col]=X[col].astype(str)

        return X
    
class ChangeTargetToNum(BaseEstimator,TransformerMixin):
    def __init__(self,target_col=None,target_val=None):
        self.target_col=target_col
        self.target_val=target_val

    def fit(self,X,y=None):

        return self
    
    def transform(self,X):
        X=X.copy()
        X.loc[X[self.target_col] != self.target_val,self.target_val] = 0
        X.loc[X[self.target_col] == self.target_val,self.target_val] = 1

        return X

class LogTransforms(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables=variables

    def fit(self,X,y=None):

        return self
    
    def transform(self,X):
        X=X.copy()
        for col in self.variables:
            X[col]=np.log()

        return X
    
class CustomLabelEncoder(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables=variables

    def fit(self,X,y=None):
        self.label_dict={}
        for var in self.variables:
            t=X[var].value_counts().sort_values(ascending=True).index
            self.label_dict[var]={k:i for i,k in enumerate(t,0)}

        return self
    
    def transform(self,X):
        X=X.copy()
        for feature  in self.variables:
            X[feature]=X[feature].map(self.label_dict[feature])

        return X