import pandas as pd

f1 = pd.read_csv("./data/Kepler.csv")
f2 = pd.read_csv("./data/TESS.csv")
final = pd.concat([f1, f2], ignore_index=True)
final.to_csv("./data/merged.csv", index=False)