from sklearn.pipeline import Pipeline
from prediction_model.config import config
import prediction_model.processing.preprocessing as pp
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LogisticRegression 


classification_pipeline=Pipeline(
    [
        #('change_tostr',pp.ChangetoSTR(variables_to_change=config.FEATURES)),
        #('changeTargetToNum',pp.ChangeTargetToNum(target_col=config.TARGET,target_val=config.TARGET_Val)),
        ('scaler',StandardScaler()),
        ('model',LogisticRegression(random_state=42))

    ]
)
