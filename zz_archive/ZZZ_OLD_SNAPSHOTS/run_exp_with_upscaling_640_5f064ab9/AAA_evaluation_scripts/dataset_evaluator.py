import numpy as np
import pandas as pd


print("#********************************** Start dataset evaluation **********************************")
print("VALIDATION SET")
df = pd.read_csv('/home/fkirchhofer/data/CheXpert-v1.0/valid.csv')
df_tasks = df.iloc[:, 5:]
counts = df_tasks.sum()
print(f"Shape of Dataframe: {df.shape}")
print(counts)

print("\n\nTEST SET")
df = pd.read_csv('/home/fkirchhofer/data/CheXpert-v1.0/test.csv')
df_tasks = df.iloc[:, 1:]
counts = df_tasks.sum()
print(f"Shape of Dataframe: {df.shape}")
print(counts)


print("#********************************** Finished dataset evaluation **********************************")