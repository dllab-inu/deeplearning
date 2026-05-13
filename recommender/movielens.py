#%%
import os
import io
import zipfile
import urllib.request
import pandas as pd
#%%
### data download
url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
with urllib.request.urlopen(url) as response:
    zip_data = response.read()
with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
    z.extractall(os.path.dirname("./data/"))
#%%
df = pd.read_csv(
    "./data/ml-1m/ratings.dat",
    sep="::",
    engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"],
    encoding="latin-1",
)
df = df[["user_id", "movie_id", "rating", "timestamp"]]
df.head()
#%%
### 원래 ID가 연속적이지 않아 이를 연속적인 번호로 변환
user2idx = {u: i for i, u in enumerate(df["user_id"].unique())}
movie2idx = {m: i for i, m in enumerate(df["movie_id"].unique())}
df["user_id"] = df["user_id"].map(user2idx)
df["movie_id"] = df["movie_id"].map(movie2idx)

n_users = df["user_id"].nunique()
n_items = df["movie_id"].nunique()
print(f"Users: {n_users}, Items: {n_items}, Interactions: {len(df)}")
#%%
df.to_csv("./data/movielens.csv")
#%%