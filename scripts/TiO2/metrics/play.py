import pandas as pd
import ast

# Load the CSV
df = pd.read_csv("/home/g15farris/bin/bayesaenet/scripts/metrics/uq_metrics_summary.csv")


for i in range(len(df)):
    for x in df.iloc[i]:
        for y in x:
            print(x, y)
    break
    