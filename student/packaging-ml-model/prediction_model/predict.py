import numpy as np
import pandas as pd
import joblib
from prediction_model.config import config
from prediction_model.processing.data_handeling import load_pipeline,load_dataset

classification_pipeline=load_pipeline(config.MODEL_NAME)

def generate_predictions(data_input):
    data=pd.DataFrame(data_input)
    pred=classification_pipeline.predict(data.FEATURES)
    output=np.where(pred==1,'y','n')
    result={'predictions':'output'}
    return result

def generate_predictions2():
    test_data=load_dataset(config.TEST_FILE)
    #data=pd.DataFrame(test_data)
    pred=classification_pipeline.predict(test_data[config.FEATURES])
    output=np.where(pred==1,'y','n')
    print(output)
    #result={'predictions':'output'}
    return output




if __name__=='__main__':
    generate_predictions2()