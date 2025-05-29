import os 
import numpy as np
import pandas as pd
import json
import pickle
from sklearn.metrics import accuracy_score,precision_score,recall_score, roc_auc_score
import logging

# ensure the logs dir exists

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger()
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(file_path:str):
    try:
        with open(file_path,'rb') as file:
            model = pickle.load(file)
        logger.debug("loaded model from %s", file_path)
        return model
    
    except pd.errors.ParserError as e:
        logger.error("failed to parse the csv file %s",e)
        raise

    except Exception as e:
        logger.error("unexpected error ocured while loading the data %s", e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(clf,x_test:np.ndarray, y_test:np.ndarray)->dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred=clf.predict(x_test)
        y_pred_proba = clf.predict_proba(x_test)[:,1]

        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        auc = roc_auc_score(y_test,y_pred_proba)

        metrics_dict= {
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'auc':auc
        }
        logger.debug("model evaluation metrics calculated")
        return metrics_dict
    except Exception as e:
        logger.debug("error during model evaluation %s",e)
        raise

def save_metrics(metrics: dict,file_path:str)->None:
    """save the evaluation metrics to a JSON file."""
    try:
        # ensure the dictionary exists
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,'w') as file:
            json.dump(metrics,file,indent=4)
        logger.debug("metrics saved to %s", file_path)

    except Exception as e:
        logger.error("error occured while saving the metrics %s",e)
        raise

def main():
    try:
        clf = load_model("model/model.pkl")
        test_data = load_data("./data/processed/test_tfidf.csv")

        x_test = test_data.iloc[:,:-1].values
        y_test = test_data.iloc[:,-1].values

        metrics = evaluate_model(clf,x_test,y_test)

        save_metrics(metrics, 'reports/metrics.json')

    except Exception as e:
        logger.error("failed to coplete model evaluation process %s",e)
        raise

if __name__ == '__main__':
    main()