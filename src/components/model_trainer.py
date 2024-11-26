import os,sys
from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_models,save_json_object,save_model
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

@dataclass
class ModelTrainerConfig():
    model_path = os.path.join('src/models','model.pkl')
    model_report_path = os.path.join('.','models_report.json')

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            x_train,y_train,x_test,y_test = (train_array[:,:-1],train_array[:,-1],
                                             test_array[:,:-1],test_array[:,-1])

            models = {
                "Decision Tree": DecisionTreeClassifier(),
                "AdaboostClassifier":AdaBoostClassifier(),
                "Gradient Boosting Classifier":GradientBoostingClassifier(verbose=1),
                "Random Forest Classifier":RandomForestClassifier(verbose=1),
                "Support Vector Machine":SVC(verbose=True,probability=True),
                "K Nearest Neighbours":KNeighborsClassifier(),
                "Naive Bayes":GaussianNB(),
                "Catboost Classifier":CatBoostClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "XGBoost Classifier":XGBClassifier()
            }
            params = {
                'Logistic Regression':{
                    'penalty':['elasticnet','l1','l2'],
                    'C': [0.1, 1, 10],                     # Regularization strength
                    'solver': ['saga', 'liblinear']        # Compatible solvers for elasticnet, l1, l2  
                },
                'Decision Tree':{
                    'max_depth':[10,20,30],
                    'min_samples_split':[2,5,10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'AdaboostClassifier':{
                    'n_estimators':[50,100,150,200],
                    'learning_rate':[0.1,0.01,0.001]
                },
                'Gradient Boosting Classifier':{
                    'n_estimators':[100,150,200],
                    'max_depth':[3,5,10],
                    'learning_rate':[0.1,0.01,0.001]
                },
                'Random Forest Classifier':{
                    'n_estimators': [100, 200, 450],       # Broader range of estimators
                    'max_features': ['sqrt', 'log2'],      # Added 'sqrt' (default for classification)
                    'max_depth': [50, 100, 340, None],     # Broader range and added None
                    'min_samples_split': [2, 3, 5],        # Common starting values
                    'min_samples_leaf': [1, 5, 10],        # Added smaller values for leaf nodes
                    'criterion': ['gini', 'entropy']       # Added entropy as an option
                },
                'Support Vector Machine':{
                    'kernel': ['linear', 'poly', 'sigmoid', 'rbf'],
                    'gamma': ['scale', 'auto'],
                    'C': [0.1, 1, 10]                      # Regularization parameter
                },
                'K Nearest Neighbours':{
                    'n_neighbors': [3, 5, 7, 10],          # Added common k values
                    'metric': ['euclidean', 'manhattan']   # Added manhattan for variety    
                },
                'Naive Bayes':{
                    'var_smoothing': [1e-9, 1e-8, 1e-7]    # Common range for variance smoothing
                },
                'Catboost Classifier':{
                    'iterations': [100, 200, 300],         # Number of boosting rounds
                    'learning_rate': [0.1, 0.01, 0.001],
                    'depth': [6, 8, 10],                   # Typical values for CatBoost depth
                    'l2_leaf_reg': [1, 3, 5]               # Regularization parameter
                },
                "XGBoost Classifier":{
                    'n_estimators': [100, 150, 200],       # Reasonable boosting rounds
                    'learning_rate': [0.1, 0.01, 0.001],
                    'max_depth': [3, 6, 10],               # Depth similar to Gradient Boosting
                    'colsample_bytree': [0.6, 0.8, 1.0],  # Fraction of features for boosting
                    'subsample': [0.8, 1.0]                # Fraction of samples for boosting
                }
            }

            model_report:dict = evaluate_models(x_train,y_train,x_test,y_test,models,params)
            best_model_name = max(model_report,key=lambda name: model_report[name]["test_accuracy"])
            best_model_score = model_report[best_model_name]["test_accuracy"]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found",sys)


            save_model(file_path=self.model_trainer_config.model_path,obj=best_model)

            predicted = best_model.predict(x_test)
            acc = accuracy_score(y_test,predicted)
            logging.info(f"best model : {best_model_name} on both training and testing data with accuracy {acc}")
            save_json_object(file_path=self.model_trainer_config.model_report_path,obj=model_report)
            return(predicted,acc,best_model_name)
        except Exception as e:
            raise CustomException(e,sys)
