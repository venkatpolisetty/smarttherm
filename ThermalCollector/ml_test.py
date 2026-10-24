import pandas as pd

df = pd.read_csv("telemetry.csv")

print(df.head())
print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum())