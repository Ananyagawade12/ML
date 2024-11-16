import pandas as pd
import numpy as np
df = pd.read_csv('filled_missing_val_data.csv')
indep_vari = ['temperature', 'humidity', 'pressure', 'rain', 'lightavg', 'lightmax', 'moisture']
for i in range(len(indep_vari)):
    col = df[indep_vari[i]]
    mean = np.mean(col)
    std_dev = np.std(col)
    print("before normalisation mean : ",mean," and stdandard dev : ",std_dev)
    #normalise
    for j in range(len(col)):
        normalised_val = (col[j]-mean)/std_dev    
        df.at[j,indep_vari[i]] = normalised_val
    new_mean = np.mean(col)
    new_stddev = np.std(col)
    print("after normalisation mean : ",new_mean," and stdandard dev : ",new_stddev)
print(df)

df.to_csv('normalised_data_part2.csv')