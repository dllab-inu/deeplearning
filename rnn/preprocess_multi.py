#%%
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
#%%
### 데이터 불러오기
df = pd.read_csv("./data/jena_climate_2009_2016.csv")
print(df.shape)
print(df.head())
#%%
### Date Time을 datetime 타입으로 변환
df["Date Time"] = pd.to_datetime(
    df["Date Time"],
    format="%d.%m.%Y %H:%M:%S"
)
#%%
### 최근 3년간의 데이터만 추출
start_date = "2014-01-01 00:00:00"
end_date   = "2016-12-31 23:59:59"

target_df = df.loc[
    (start_date <= df["Date Time"]) & (df["Date Time"] <= end_date)
]

print(target_df.shape)
print(target_df.head())
#%%
### 시간별 관측 데이터 추출
target_df_hour = target_df.loc[
    (target_df["Date Time"].dt.minute == 0) & (target_df["Date Time"].dt.second == 0)
]
target_df_hour.to_csv("./data/climate_multi.csv")

print(target_df_hour.shape)
print(target_df_hour.head())
#%%