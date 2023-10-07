import os,sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from src.utils import save_model
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


@dataclass
class DataTransformationConfig():
    preprocessor_file_path = os.path.join('src/models','preprocessor.pkl')
    cat_preprocessor_file_path = os.path.join('src/models','cat_preprocessor.pkl')

class DataTransformation():
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self,trainset_path,testset_path):
        try:
            train_df = pd.read_csv(trainset_path)
            test_df = pd.read_csv(testset_path)

            logging.info("Reading the train and test data completed.")
            train_df=train_df.drop(["Unnamed: 133"],axis=1)
            test_df = test_df.drop(["Unnamed: 133"],axis=1)
            logging.info("Obtaining the preprocessor object.")

            preprocessing_obj = StandardScaler()
            cat_preprocessing_obj = LabelEncoder()
            target_column_name = "prognosis"

            logging.info("Splitting the features and target column.")

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on both training and testing object.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            target_feature_train_df = cat_preprocessing_obj.fit_transform(target_feature_train_df)
            target_feature_test_df = cat_preprocessing_obj.transform(target_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            logging.info("Data Transformation complete now saving the preprocessing object")
            save_model(self.data_transformation_config.preprocessor_file_path,preprocessing_obj)
            save_model(self.data_transformation_config.cat_preprocessor_file_path,cat_preprocessing_obj)
            return(
                train_arr,test_arr
            )
        except Exception as e:
            raise CustomException(e,sys)
