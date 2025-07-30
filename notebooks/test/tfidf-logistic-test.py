# %%
import os
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from joblib import load

# %% run once
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# %%
def clean_text(text):
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^a-z\s]', '', text)

    return text

# %%
def apply_additional_processing(text):
    text = str(text)
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)

# %%
# --- 3. 통합 전처리 파이프라인 함수 ---
def full_text_preprocessing_pipeline(text):
    """
    Applies the full preprocessing pipeline (clean_text + additional processing).
    This mimics the exact preprocessing done before TF-IDF vectorization in training.
    """
    # 1단계: clean_text 적용
    cleaned_step1 = clean_text(text)
    # 2단계: apply_additional_processing 적용
    final_processed_text = apply_additional_processing(cleaned_step1)
    return final_processed_text

# %%
# --- 4. 예측 함수 정의 ---
def predict_sentiment(text_list, model, vectorizer, preprocess_pipeline_func):
    """
    주어진 텍스트 리스트에 대해 감성을 예측합니다.

    Args:
        text_list (list): 예측할 텍스트 문자열의 리스트입니다.
        model: 학습된 감성 분류 모델 (예: LogisticRegression).
        vectorizer: 텍스트를 벡터화할 도구 (예: TfidfVectorizer).
        preprocess_pipeline_func: 전체 전처리 파이프라인 함수 (full_text_preprocessing_pipeline 사용).

    Returns:
        list: 각 텍스트에 대한 예측된 감성 레이블 (0: 부정, 1: 긍정) 리스트.
    """
    # 새로운 입력 텍스트에 전체 전처리 파이프라인 적용
    processed_texts = [preprocess_pipeline_func(text) for text in text_list]
    print(f"전처리된 텍스트: {processed_texts}")

    # 학습 시 사용했던 vectorizer를 사용하여 transform만 수행해야 합니다.
    text_vectors = vectorizer.transform(processed_texts)

    # 감성 예측
    predictions = model.predict(text_vectors)

    # 예측 결과를 읽기 쉽게 변환
    sentiment_labels = ["부정 (0)" if p == 0 else "긍정 (1)" for p in predictions]

    return sentiment_labels

# %%
# --- 5. 인터랙티브 테스트 함수 정의 ---
def interactive_sentiment_test(model, vectorizer, preprocess_pipeline_func):
    """
    터미널에서 사용자 입력을 받아 실시간으로 감성 분류를 수행합니다.
    """
    print("\n--- 실시간 감성 분류 테스트를 시작합니다 ---")
    print("분석하고 싶은 문장을 입력하고 Enter를 누르세요.")
    print("종료하려면 'quit' 또는 'exit'을 입력하세요.")

    while True:
        user_input = input("\n> 문장을 입력하세요: ")

        if user_input.lower() in ['quit', 'exit']:
            print("테스트를 종료합니다.")
            break

        if not user_input.strip(): # 빈 문자열 입력 처리
            print("입력된 문장이 없습니다. 다시 시도해주세요.")
            continue

        try:
            predicted_sentiments = predict_sentiment(
                [user_input], # 단일 입력을 리스트 형태로 전달
                model,
                vectorizer,
                preprocess_pipeline_func # 통합 전처리 파이프라인 함수 전달
            )
            print(f"-> 예측 감성: {predicted_sentiments[0]}")

        except Exception as e:
            print(f"오류 발생: {e}. 다시 시도해주세요.")


# %%
if __name__ == "__main__":
    print("실시간 감성 분류 테스트를 위한 모델 및 벡터라이저 로드 시작...")

    # interactive_test.py 파일의 현재 경로를 가져옵니다.
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 프로젝트 루트 디렉토리를 계산합니다.
    project_root = os.path.dirname(os.path.dirname(current_script_dir))

    # 모델이 project_root/models/ 에 바로 있다고 가정합니다.
    model_dir = os.path.join(project_root, 'models')
    
    # 모델 파일과 벡터라이저 파일의 전체 경로를 지정합니다.
    logistic_regression_model_path = os.path.join(model_dir, 'tfidf_logistic_regression_model.joblib')
    tfidf_vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.joblib')

    logistic_regression_model = None
    tfidf_vectorizer = None

    try:
        # 저장된 모델과 벡터라이저 로드
        logistic_regression_model = load(logistic_regression_model_path)
        tfidf_vectorizer = load(tfidf_vectorizer_path)
        print("모델과 벡터라이저가 성공적으로 로드되었습니다.")

        interactive_sentiment_test(
            logistic_regression_model,
            tfidf_vectorizer,
            full_text_preprocessing_pipeline # 통합 전처리 파이프라인 함수 전달
        )
    except FileNotFoundError:
        print(f"오류: 모델 파일이나 벡터라이저 파일을 찾을 수 없습니다.")
        print(f"경로를 확인하거나, 'tfidf-logistic-regression.py'를 실행하여 모델을 학습하고 저장하세요.")
        print(f"예상 모델 경로: {logistic_regression_model_path}")
        print(f"예상 벡터라이저 경로: {tfidf_vectorizer_path}")
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")
# %%
