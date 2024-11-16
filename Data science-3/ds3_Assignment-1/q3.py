# q3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('filled_missing_val_data.csv')
indep_vari = ['temperature', 'humidity', 'pressure', 'rain', 'lightavg', 'lightmax', 'moisture']

# boxplots
for i in range(len(indep_vari)):
    plt.boxplot(df[indep_vari[i]])
    plt.title('Boxplot for Detecting Outliers of '+ str(indep_vari[i]))
    plt.xlabel(indep_vari[i])
    plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
    plt.grid(True)
    plt.show()

outlier_dict = {"temperature":[],"humidity":[],"pressure":[],"rain":[],"lightavg":[],"lightmax":[],"moisture":[]}
for i in range(len(indep_vari)):
    col = indep_vari[i]
    sorted_df = df.sort_values(by=col)
    n = len(sorted_df[col])
    if n % 2 == 1:
        median = sorted_df[col][n // 2]
    else:
        median = (sorted_df[col][n // 2 - 1] + sorted_df[col][n // 2]) / 2

    q1 = sorted_df[col].quantile(0.25)              #25th percentile
    q3 = sorted_df[col].quantile(0.75)              #75th percentile
    IQR = q3 - q1

    for j in range(len(sorted_df[col])) :
        attribute_value = sorted_df[col][j]
        if not (q1 - (1.5*IQR)) < attribute_value < (q3 + (1.5*IQR)):
            #hence outlier
            outlier_dict [col].append(attribute_value)
            df.at[j,col] = median
df.to_csv("filled_missing_val_data.csv")
print(outlier_dict)
# boxplots
for i in range(len(indep_vari)):
    plt.boxplot(df[indep_vari[i]])
    plt.title('Boxplot of '+ str(indep_vari[i])+'after replacing outliers with median')
    plt.xlabel(indep_vari[i])
    plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
    plt.grid(True)
    plt.show()