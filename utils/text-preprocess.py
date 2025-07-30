# preprocess_data.py

# %%
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# %%
# NLTK data downloads(run this once)
nltk.download('stopwords')
nltk.download('wordnet')

# %%
def clean_text(text):
    """cleaning text by removing unwanted characters and formatting"""
    # 1. lowercase
    text = text.lower()
    # 2. eliminate mentions (@username)
    text = re.sub(r'@\w+', '', text)
    # 3. remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # 4. remove hashtags
    text = re.sub(r'#\w+', '', text)
    # 5. remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # 6. remove special characters
    # this can vary based on the model.
    # for now, we will keep only alphabets and spaces.
    text = re.sub(r'[^a-z\s]', '', text)

    return text


# %%
def preprocess_and_save_data(input_csv_path, output_csv_path, sample_size=20000):
    """
    Loads the dataset, preprocesses it, and saves the cleaned data.
    """
    print(f"Loading dataset: {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path,
                         encoding='ISO-8859-1',
                         names=['sentiment', 'id', 'date', 'query', 'user', 'text'])
    except Exception as e:
        print(f"error with data load: {e}. trying with latin-1 encoding.")
        df = pd.read_csv(input_csv_path,
                         encoding='latin-1',
                         names=['sentiment', 'id', 'date', 'query', 'user', 'text'])

    df = df[['sentiment', 'text']]
    df['sentiment'] = df['sentiment'].replace(4, 1) # positive 4 -> 1

    # dataset size limit
    if sample_size < len(df):
        print(f"Sampling dataset: {sample_size}.")
        df_positive = df[df['sentiment'] == 1].sample(n=sample_size // 2, random_state=25)
        df_negative = df[df['sentiment'] == 0].sample(n=sample_size // 2, random_state=25)
        df = pd.concat([df_positive, df_negative]).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        print(f"dataset size: {len(df)}")

    print("Preprocessing text...")
    df['cleaned_text'] = df['text'].apply(clean_text)

    # additional processing and lemmatization(optional)
    # this can be skipped for deep learning models.
    # simple example of additional processing:

    # stop_words = set(stopwords.words('english'))
    # lemmatizer = WordNetLemmatizer()
    # def apply_additional_processing(text):
    #     words = text.split()
    #     words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    #     return ' '.join(words)
    #
    # df['cleaned_text'] = df['cleaned_text'].apply(apply_additional_processing)
    # print("additional processing completed.")

    # remove empty cleaned_text
    df = df[df['cleaned_text'].str.strip() != '']

    print(f"size of processed and sampled dataset: {len(df)}")
    print("Top 5 processed data:")
    print(df.head())

    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"saved processed data to '{output_csv_path}'")

# %%
if __name__ == "__main__":
    # dir path
    dir_path = '../data'

    input_path = f'{dir_path}/training.1600000.processed.noemoticon.csv' # original data path
    output_path = f'{dir_path}/preprocessed_sentiment_data_20k.csv' # output data path after preprocessing

    preprocess_and_save_data(input_path, output_path, sample_size=20000)
# %%
