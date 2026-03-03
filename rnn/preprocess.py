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
### 예측하고자 하는 변수 (단변량): 기온
df = df[["Date Time", "T (degC)"]] # [T, ]
print(df.shape)
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
target_df_hour.to_csv("./data/climate.csv")

print(target_df_hour.shape)
print(target_df_hour.head())
#%%
### 시각화
plt.figure(figsize=(15, 5))
plt.plot(
    target_df_hour["Date Time"].iloc[-1000:], target_df_hour["T (degC)"].iloc[-1000:],
    linewidth=2
)
plt.xlabel("timesteps", fontsize=18)
plt.ylabel("Temperature (degC)", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("Daily Average Temperature", fontsize=19)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./fig/observations.png")
plt.show()
plt.close()
#%%