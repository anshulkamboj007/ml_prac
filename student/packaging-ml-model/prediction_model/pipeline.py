from sklearn.pipeline import Pipeline
from prediction_model.config import config
import prediction_model.processing.preprocessing as pp


classification_pipeline=Pipeline(
    [
        ('mean_imputation',pp.MeanImputer(variables=config.NUM_FEATURES)),
        ('change_tostr',pp.ChangetoSTR(variables_to_change=config.CAT_FEATURES)),
        ('mode_imputation',pp.ModeImputer(variables=config.CAT_FEATURES)),
        ('changeTargetToNum',pp.ChangeTargetToNum(target_col=config.TARGET,target_val=config.TARGET_Val))

    ]
)
