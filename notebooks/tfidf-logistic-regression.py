# %%
import pandas as pd
import os

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from joblib import dump

# %% run once
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# %%
def apply_additional_processing(text):
    """
    기본 전처리가 완료된 텍스트에 대해 불용어 제거 및 표제어 추출을 수행합니다.
    (preprocess_data.py의 clean_text 함수가 이미 적용된 텍스트에 사용)
    """
    text = str(text) # 입력이 문자열인지 확인
    tokens = text.split()
    # 불용어 제거 및 표제어 추출
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)

# %%
def load_and_split_data(data_csv_path, test_size=0.2, random_state=25):
    """
    전처리된 데이터를 로드하고, 추가 전처리를 수행한 뒤, 학습/테스트 세트로 분리합니다.
    """
    if not os.path.exists(data_csv_path):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_csv_path}")

    print(f"전처리된 데이터 로드 중: {data_csv_path}")
    df = pd.read_csv(data_csv_path, encoding='utf-8')

    print("로드된 데이터 상위 5개:")
    print(df.head())
    print("\n감성 분포:")
    print(df['sentiment'].value_counts())

    if 'cleaned_text' not in df.columns:
        raise ValueError("데이터프레임에 'cleaned_text' 컬럼이 없습니다. preprocess_data.py를 실행하여 'cleaned_text'가 포함된 파일을 생성했는지 확인해주세요.")
    
    # 'cleaned_text' 컬럼이 문자열이 아닌 경우를 대비
    df['cleaned_text'] = df['cleaned_text'].astype(str)

    print("로드된 텍스트에 추가 전처리(불용어 제거, 표제어 추출) 적용 중...")
    df['final_processed_text'] = df['cleaned_text'].apply(apply_additional_processing)
    
    # 추가 전처리 후 빈 문자열이 된 행 제거
    df = df[df['final_processed_text'].str.strip() != '']
    
    if df.empty:
        raise ValueError("모든 전처리 후 데이터프레임이 비어 있습니다. 전처리 과정을 확인해주세요.")

    print(f"최종 전처리된 데이터셋 크기: {len(df)}개")
    print("최종 전처리된 데이터 상위 5개:")
    print(df.head())

    X = df['final_processed_text'] # 최종 전처리된 텍스트 사용
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    print(f"학습 데이터셋 크기: {len(X_train)}개")
    print(f"테스트 데이터셋 크기: {len(X_test)}개")

    return X_train, X_test, y_train, y_test

# %%
# TF-IDF vectorization
def vectorize_text(X_train, X_test, max_features=5000):
    print("TF-IDF 벡터화 중...")
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"학습 데이터 TF-IDF 벡터 크기: {X_train_tfidf.shape}")
    print(f"테스트 데이터 TF-IDF 벡터 크기: {X_test_tfidf.shape}")
    return X_train_tfidf, X_test_tfidf, vectorizer

# %%
# Train and evaluate the model
def train_and_evaluate_model(X_train_tfidf, y_train, X_test_tfidf, y_test, max_iter=1000, random_state=25):
    print("로지스틱 회귀 모델 학습 중...")
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(X_train_tfidf, y_train)

    print("모델 예측 중...")
    y_pred = model.predict(X_test_tfidf)

    print("모델 평가:")
    print(classification_report(y_test, y_pred))
    print(f"정확도: {accuracy_score(y_test, y_pred):.4f}")

    return model, y_pred

# %%
def save_model_and_vectorizer(model, vectorizer, model_path, vectorizer_path):
    """
    학습된 모델과 TF-IDF 벡터라이저를 지정된 경로에 저장합니다.

    Args:
        model: 저장할 학습된 머신러닝 모델.
        vectorizer: 저장할 TF-IDF 벡터라이저 객체.
        model_path (str): 모델을 저장할 파일 경로.
        vectorizer_path (str): 벡터라이저를 저장할 파일 경로.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True) # 모델 저장 폴더가 없으면 생성
    os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True) # 벡터라이저 저장 폴더가 없으면 생성

    dump(model, model_path)
    print(f"모델이 '{model_path}'에 저장되었습니다.")

    dump(vectorizer, vectorizer_path)
    print(f"벡터라이저가 '{vectorizer_path}'에 저장되었습니다.")


# %%
if __name__ == "__main__":
    preprocessed_data_path = '../data/preprocessed_sentiment_data_20k.csv'
    load_and_split_data(preprocessed_data_path)

    print("--- 1. 데이터 로드 및 분리 ---")
    X_train, X_test, y_train, y_test = load_and_split_data(preprocessed_data_path)

    print("\n--- 2. 텍스트 벡터화 (TF-IDF) ---")
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = vectorize_text(X_train, X_test)

    print("\n--- 3. 모델 학습 및 평가 ---")
    logistic_regression_model, y_predicted = train_and_evaluate_model(X_train_tfidf, y_train, X_test_tfidf, y_test)

    print("\n--- 4. 모델과 벡터라이저 저장 ---")
    logistic_regression_model_path = '../models/tfidf_logistic_regression_model.joblib'
    tfidf_vectorizer_path = '../models/tfidf_vectorizer.joblib'
    save_model_and_vectorizer(
        logistic_regression_model,
        tfidf_vectorizer,
        logistic_regression_model_path,
        tfidf_vectorizer_path
    )

    print("\n모든 TF-IDF + 로지스틱 회귀 모델 워크플로우 완료.")


# %%
