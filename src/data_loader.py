# data_loader.py
import pandas as pd
import logger_setup
from conf.config import RAW_DATA_FILE_01, RAW_DATA_FILE_02, RAW_DATA_FILE_03

class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self):
        try:
            logger_setup.logger.debug("START ...")
            logger_setup.logger.debug(f'Loading data from path: {self.data_path}')
            df_train = pd.read_csv(f'{self.data_path}{RAW_DATA_FILE_01}')
            df_test = pd.read_csv(f'{self.data_path}{RAW_DATA_FILE_02}')
            df_data_dict = pd.read_csv(f'{self.data_path}{RAW_DATA_FILE_03}')
            logger_setup.logger.info(f'Loaded {RAW_DATA_FILE_01} with shape: {df_train.shape}')
            logger_setup.logger.info(f'Loaded {RAW_DATA_FILE_02} with shape: {df_test.shape}')
            logger_setup.logger.info(f'Loaded {RAW_DATA_FILE_03} with shape: {df_data_dict.shape}')
            logger_setup.logger.info(f'Successfully loaded data from path: {self.data_path}')
            logger_setup.logger.debug("... FINISH")
            return df_train, df_test, df_data_dict
        except Exception as e:
            logger_setup.logger.error(f'Failed to load data from {self.data_path}: {str(e)}')
            raise