import pandas as pd
import ast

df = pd.read_csv("embeddings.csv")
embeddings = df["vectors"].apply(lambda x: list(map(float, ast.literal_eval(x)))).to_list()
print(embeddings)