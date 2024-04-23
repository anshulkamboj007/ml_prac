
import pytest
from prediction_model.config import config
from prediction_model.processing.data_handeling import load_dataset

from prediction_model.predict import generate_predictions

#op not null
#op is str type

@pytest.fixture
def single_prediction():
    test_dataset=load_dataset(config.TEST_FILE)
    single_row=test_dataset[:1]
    result=generate_predictions(single_row)
    print(result)
    return result

def test_single_pred_not_none(single_prediction):
    assert single_prediction is not None

def test_single_pred_str_type(single_prediction):
    assert isinstance(single_prediction.get('predictions')[0],str)

def test_single_pred_validate(single_prediction):
    assert single_prediction.get('predictions')[0]=='y'