import os
import sys
sys.path.append('src')
from exception import CustomException

from logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

from components.data_transformation import DataTransformation
from components.data_transformation import DataTransformationConfig

from components.model_trainer import ModelTrainerConfig
from components.model_trainer import ModelTrainer



@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts',"train.csv")
    test_data_path: str  = os.path.join('artifacts',"test.csv")
    raw_data_path: str   = os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered into data ingestion")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the dataset as dataframe")

            #os.makedirs(os.path.join(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            df.columns= df.columns.str.replace(" ","_")
            df.columns= df.columns.str.replace("/","_")

            logging.info("Train test split initiated")

            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)


if __name__=='__main__':
    obj=DataIngestion()
    train_path,test_path=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()

    train_array,test_array,preprocessing_path = data_transformation.initiate_data_transformation(train_path,test_path)

    model_trainer= ModelTrainer()

    r2_score=model_trainer.initiate_model_trainer(train_array,test_array,preprocessing_path)

    logging.info("r2_score of the best model is {}".format(r2_score))





        

        




