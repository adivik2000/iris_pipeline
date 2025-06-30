import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")
df = df.sample(frac=0.8, random_state=42)
df.to_csv("data/iris.csv", index=False)