import pandas as pd
import numpy as np

df = pd.read_csv('landslide_data_miss.csv')
df_cleaned = df.dropna(subset=['stationid'])
threshold = df_cleaned.shape[1] * (1/3)
df_cleaned = df_cleaned.dropna(thresh=df_cleaned.shape[1] - threshold + 1)
df_cleaned.to_csv('cleaned_landslide_data.csv', index=False)

