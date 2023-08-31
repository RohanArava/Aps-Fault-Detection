from sklearn.metrics import f1_score
from sensor import utils
from sensor.entity import config_entity, artifact_entity
from sensor.exception import SensorException
from sensor.logger import logging
import sys
from xgboost import XGBClassifier

class ModelTrainer:
    def __init__(self, model_trainer_config:config_entity.ModelTrainerConfig, data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)


    def train_model(self, x, y):
        xgb_clf = XGBClassifier()
        xgb_clf.fit(x, y)
        return xgb_clf
    
    def initiate_model_trainer(self) -> artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            x_train, y_train = train_arr[:,:-1], train_arr[:,-1]
            x_test, y_test = test_arr[:,:-1], test_arr[:,-1]

            model = self.train_model(x=x_train, y=y_train)
            
            yhat_train = model.predict(x_train)
            f1_train_score = f1_score(y_true=y_train, y_pred = yhat_train)

            yhat_test = model.predict(x_test)
            f1_test_score = f1_score(y_true=y_test, y_pred=yhat_test)

            logging.info(f"f1_teat_score: {f1_test_score}, f1_train_scor: {f1_train_score}")
            if f1_test_score < self.model_trainer_config.expected_score:
                raise Exception(f"Model does not give expected accuracy\nExpected:{self.model_trainer_config.expected_score}\nActual:{f1_test_score}")
            
            diff = abs(f1_train_score - f1_test_score)

            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(f"train and test score difference: {diff} > Overfitting threshold: {self.model_trainer_config.overfitting_threshold}")
            
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            model_trainer_artifact =  artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path,
                                                        f1_test_score=f1_test_score, f1_train_score=f1_train_score)
            logging.info(f"Model trainer_artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise SensorException(e, sys)
        
        