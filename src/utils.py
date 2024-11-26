import os,sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
import dill,json
from src.logger import logging
from src.exception import CustomException

def save_model(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as f:
            dill.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(x_train,y_train,x_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            logging.info(f"Evaluation initiated for {model}.")
            para = param[list(models.keys())[i]]
            rs = RandomizedSearchCV(estimator=model,param_distributions=para,cv=3,n_iter=9,random_state=100,n_jobs=-1)
            logging.info(f"RandomizedSearchCV initiated for {model}.")
            rs.fit(x_train,y_train)
            logging.info(f"RandomizedSearchCV fit done and set_params initiated for {model}.")
            model.set_params(**rs.best_params_)
            logging.info(f"setting parameters completed and fitting initiated for {model}.")
            model.fit(x_train,y_train)
            logging.info(f"prediction initiated for {model}.")
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            logging.info(f"Getting the accuracy for train and test data for {model}")
            train_model_accuracy = accuracy_score(y_true=y_train,y_pred=y_train_pred)
            test_model_accuracy = accuracy_score(y_true=y_test,y_pred=y_test_pred)
            precision = precision_score(y_true=y_test, y_pred=y_test_pred, average='macro',zero_division=0)
            recall = recall_score(y_true=y_test, y_pred=y_test_pred, average='macro')
            f1 = f1_score(y_true=y_test, y_pred=y_test_pred, average='macro')
            roc_auc = roc_auc_score(y_true=y_test, y_score=model.predict_proba(x_test),multi_class="ovr")
            report[list(models.keys())[i]] = {
                "test_accuracy": test_model_accuracy,
                "train_accuracy": train_model_accuracy,  # Ensure correct key matching
                "accuracy_variance": train_model_accuracy - test_model_accuracy,  # For overfitting/underfitting check
                "precision": precision,  # Classification metric
                "recall": recall,        # Classification metric
                "f1_score": f1,          # Classification metric
                "roc_auc_score": roc_auc,  # Useful for binary classification
            }
            logging.info(f"Obtained accuracy of {test_model_accuracy} and completed with {model}.")
        return report
    except Exception as e:
        raise CustomException(e,sys)

def save_json_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'w') as f:
            json.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)

