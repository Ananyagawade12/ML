#2 part b RMSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df_original = pd.read_csv('landslide_data_original.csv')
df_new = pd.read_csv('filled_missing_val_data.csv')

indep_vari = [ "temperature", "humidity", "pressure", "rain", "lightavg", "lightmax", "moisture"]
rmse_list = [0,0,0,0,0,0,0]
j = i = 0
N = len(df_new)
for i in range(N):
    if df_original["dates"][j] != df_new["dates"][i]:
        while df_original["dates"][j] != df_new["dates"][i]:
            j += 1
    for k in range(len(indep_vari)):
        var = indep_vari[k]
        rmse_list[k] += ((df_new[var][i] - df_original[var][j])**2)
#print(rmse_list)
rmse_vals = [(i/N)**0.5 for i in rmse_list]
#print(rmse_vals)

plt.plot(indep_vari,rmse_vals)
plt.xlabel("attributes")
plt.ylabel("RMSE values")
plt.yticks(range(0, 1200, 200))
plt.show()