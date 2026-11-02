import pandas as pd
import numpy as np

def print_stats(name, df):
    print(f"--- Stats for {name} ---")
    features = [c for c in df.columns if c not in ['anomaly', 'changepoint', 'datetime']]
    stats = []
    for f in features:
        stats.append({
            'Feature': f,
            'Mean': df[f].mean(),
            'Std': df[f].std(),
            'Min': df[f].min(),
            'Max': df[f].max()
        })
    print(pd.DataFrame(stats).to_string(index=False))
    print("\n")

# Load valve1/0.csv
df_v1 = pd.read_csv('data/raw/skab_repo/data/valve1/0.csv', sep=';')
print_stats("valve1/0.csv (Rows 0-399, Train)", df_v1.iloc[:400])

# Load anomaly-free.csv
df_af = pd.read_csv('data/raw/skab_repo/data/anomaly-free/anomaly-free.csv', sep=';')
print_stats("anomaly-free.csv (Rows 0-399, Train)", df_af.iloc[:400])
print_stats("anomaly-free.csv (Rows 400-1147, Stream)", df_af.iloc[400:1147])
