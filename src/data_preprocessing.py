import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import string
import os
import logging
nltk.download('stopwords')
nltk.download('punkt_tab')


# ensure the logs dir exists

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger()
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
    """
    ps = PorterStemmer()
    # convert to lowercase
    text = text.lower()
    # tokenize the text
    text = nltk.word_tokenize(text)
    # remove non-alphanumeric values
    text = [word for word in text if word.isalnum()]
    # remove stopword and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # stem words
    text= [ps.stem(word) for word in text]
    # join the token back into the single string
    return " ".join(text)
def preprocess_df(df,text_column='text',target_column='target'):
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates, and transforming the text column.
    """
    try:
        logger.debug("starting preprocessing for dataframe")
        # encode the target column
        encoder = LabelEncoder()
        df[target_column]=encoder.fit_transform(df[target_column])
        logger.debug("target column encoded")

        # remove duplicate raw
        df = df.drop_duplicates(keep='first')
        logger.debug("text column transformed")

        # apply text transformation to the specified text column
        df.loc[:,text_column] = df[text_column].apply(transform_text)
        logger.debug("text column transformed")
        return df
    except KeyError as e:
        logger.error("column not found %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error occured while pre-processing %s",e)
        raise

def main(text_column='text',target_column='target'):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        # fetch the data from data/raw
        train_data= pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")
        logger.debug("train and test data loaded")

        # transform the data
        train_processed_data = preprocess_df(train_data,text_column,target_column)
        test_processed_data = preprocess_df(test_data,text_column,target_column)

        # store the data inside data/processed
        data_path = os.path.join('./data','interim')
        os.makedirs(data_path,exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path,'train_processed.csv'),index=False)
        test_processed_data.to_csv(os.path.join(data_path,'test_processed.csv'),index=False)

        logger.debug("processed data saved to %s", data_path)
        
    except FileNotFoundError as e:
        logger.error("file not found %s",e)

    except pd.errors.EmptyDataError as e:
        logger.error("No data %s",e)
    except Exception as e:
        logger.error("failed to complete process of data tansformation %s",e)
        print(f"errro: {e}")

if __name__ == '__main__':
    main()