import pandas as pd

df = pd.read_csv('filled_missing_val_data.csv')

#range = 5 to 12
lower_bound = 5
upper_bound = 12
indep_vari = ['temperature', 'humidity', 'pressure', 'rain', 'lightavg', 'lightmax', 'moisture']
for i in range(len(indep_vari)):
    col = df[indep_vari[i]]
    min_og = min(col)
    max_og = max(col)
    print("before normalisation min : ",min_og," and max : ",max_og)
    for j in range(len(col)):
        original_val = col[j]
        normalised_val = ((original_val - min_og)*(upper_bound-lower_bound)/(max_og-min_og)) + lower_bound
        df.at[j,indep_vari[i]] = normalised_val
    print("after normalisation min : ",min(df[indep_vari[i]])," and max : ",max(df[indep_vari[i]]))
df.to_csv("normalised_data.csv")