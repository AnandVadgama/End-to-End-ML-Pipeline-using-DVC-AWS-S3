import pandas as pd
import numpy as np
import os
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier

# ensure the logs dir exists

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger()
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str)->pd.DataFrame:
    """load data frame from a csv file
    :param file path: path to the csv file
    :return : loaded dataframe
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug("data loaded from %s and shape of %s",file_path,df.shape)
        return df
        
    except pd.errors.ParserError as e:
        logger.error("failed to parse the csv file %s", e)
        raise

    except FileNotFoundError as e:
        logger.error("file not found %s", e)
        raise

    except Exception as e:
        logger.debug("unexpected error occured while loading the data %s", e)
        raise

def train_model(x_train: np.ndarray, y_train: np.ndarray, params:dict)->RandomForestClassifier:
    """
    Train the RandomForest model.
    
    :param X_train: Training features
    :param y_train: Training labels
    :param params: Dictionary of hyperparameters
    :return: Trained RandomForestClassifier
    """
    try:
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("the number of samples in x_train and y_train must be the same")
        
        logger.debug("initializing randomforest model with classifier: %s", params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])

        logger.debug("model training started with %d samples", x_train.shape[0])
        clf.fit(x_train,y_train)
        logger.debug("model training completed")

        return clf
    except ValueError as e:
        logger.error("value error occured during training %s", e)
        raise

    except Exception as e:
        logger.error("error during model training %s",e)
        raise

def save_model(model, file_path:str)-> None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    try:
        # ensure the directory exists
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug("model saved at %s", file_path)

    except FileNotFoundError as e:
        logger.error("file not found  %s",e)
        raise
    except Exception as e:
        logger.error("erro occured while saving the model %s",e)
        raise

def main():
    try:
        params = {'n_estimators':25, 'random_state':2}
        train_data = load_data('./data/processed/train_tfidf.csv')
        x_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values

        clf = train_model(x_train,y_train,params)

        model_save_path= 'model/model.pkl'
        save_model(clf,model_save_path)

    except Exception as e:
        logger.error("failed to coplete model building process %s",e)
        raise

if __name__ == '__main__':
    main()
        