import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# ensure the logs dir exists

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger()
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str)->pd.DataFrame:
    """load data from a csv file"""
    try:
        df = pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logger.debug("data loaded and NaNs filled from %s",file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("failed to parse the csv file %s",e)
        raise
    except Exception as e:
        logger.error("Unexpeted error occured while loading data %s",e)
        raise
def apply_tfidf(train_data:pd.DataFrame,test_data:pd.DataFrame, max_features: int)->tuple:
    """apply tfidf to the data"""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        x_train=train_data['text'].values
        y_train=train_data['target'].values
        x_test=test_data['text'].values
        y_test=test_data['target'].values

        x_train_bow = vectorizer.fit_transform(x_train)
        x_test_bow = vectorizer.transform(x_test)

        train_df = pd.DataFrame(x_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(x_test_bow.toarray())
        test_df['label'] = y_test

        logger.debug("tfidf applied and transform")
        return train_df,test_df

    except Exception as e:
        logger.error("error during tfidf word transformation %s",e)
        raise

def save_data(df:pd.DataFrame, file_path:str)->None:
    """save data frame to a csv file"""
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path,index=False)
        logger.debug("data saved to %s",file_path)
    except Exception as e:
        logger.error("Unexpected error occured while saving the data %s",e)
        raise

def main():
    try:
        max_features= 50

        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        train_df, test_df = apply_tfidf(train_data,test_data,max_features)

        save_data(train_df,os.path.join('./data','processed','train_tfidf.csv'))
        save_data(test_df,os.path.join('./data','processed','test_tfidf.csv'))

    except Exception as e:
        logger.error("failed to complete the feature extraction process %s",e)
        raise

if __name__ == '__main__':
    main()

