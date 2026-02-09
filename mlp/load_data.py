#%%
import numpy as np
import pandas as pd
#%%
### 데이터 불러오기
column_description = {
    "age": "Age of the individual (integer)",
    "workclass": "Type of employment (categorical)",
    "fnlwgt": "(사용하지 않는 변수 --- 각 관측치별 가중치)",
    "education": "Highest education level achieved (categorical)",
    "education-num": "Number of years of education (integer)",
    "marital-status": "Marital status of the individual (categorical)",
    "occupation": "Occupation type (categorical)",
    "relationship": "Relationship status within household (categorical)",
    "race": "Race of the individual (categorical)",
    "sex": "Gender of the individual (categorical)",
    "capital-gain": "Capital gain in the previous year (integer)",
    "capital-loss": "Capital loss in the previous year (integer)",
    "hours-per-week": "Number of working hours per week (integer)",
    "native-country": "Country of origin (categorical)",
    "income": "Income class (target variable): <=50K or >50K"
}

train_df = pd.read_csv(
    "./data/adult.data", 
    header=None, names=column_description.keys())
test_df = pd.read_csv(
    "./data/adult.test", 
    header=None, comment="|", names=column_description.keys())
# header=None: 첫 줄부터 데이터로 처리, 열 이름은 자동으로 0,1,2,...의 숫자로 생성
# comment="|": | 로 시작하는 설명문은 제거
# names: 열 이름 설정
#%%
### 데이터 전처리
test_df["income"] = test_df["income"].astype(str).str.replace(".", "")

for c in train_df.columns:
    train_df[c] = train_df[c].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )
train_df = train_df.replace("?", np.nan).dropna().reset_index(drop=True)

for c in test_df.columns:
    test_df[c] = test_df[c].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )
test_df = test_df.replace("?", np.nan).dropna().reset_index(drop=True)

print("학습데이터 크기:", train_df.shape)
print("테스트데이터 크기:", test_df.shape)

### target 변수 생성
train_df["income"] = train_df["income"].apply(
    lambda x: 1 if x == ">50K" else 0).astype(int)
test_df["income"] = test_df["income"].apply(
    lambda x: 1 if x == ">50K" else 0).astype(int)

train_df.to_csv("./data/train.csv")
test_df.to_csv("./data/test.csv")
#%%