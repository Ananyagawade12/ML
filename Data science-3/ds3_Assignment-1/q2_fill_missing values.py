import pandas as pd
import numpy as np
from q1 import get_original_stats
df_cleaned = pd.read_csv('cleaned_landslide_data.csv')

indep_vari = ["dates", "stationid", "temperature", "humidity", "pressure", "rain", "lightavg", "lightmax", "moisture"]

for var in indep_vari:
    #print(f"Processing: {var}")
    column = df_cleaned[var] 

    for i in range(len(column)):
        if pd.isna(column.iloc[i]):  # Use iloc for positional indexing
            next_index = i + 1
            while next_index < len(column) and pd.isna(column.iloc[next_index]):
                next_index += 1
            
            if i == 0:
                df_cleaned.at[i, var] = column.iloc[next_index]
                continue
            prev_index = i - 1
            if next_index == len(column)-1 and pd.isna(column.iloc[next_index]):
                df_cleaned.at[i, var] = column.iloc[prev_index]
            
            prev_index = i - 1
            #print("preind : ",prev_index , " n ind: ",next_index)
            if prev_index >= 0 and next_index < len(column):
                prev_val = column.iloc[prev_index]
                next_val = column.iloc[next_index]
                interpolated_value = prev_val + (next_val - prev_val) * (i - prev_index) / (next_index - prev_index)
                df_cleaned.at[i, var] = interpolated_value   
            #print("var : ",var,"i: ",i,"interpolated value : ",interpolated_value)
            #print("value in df where interpolated value was assigne  ",df_cleaned[var].iloc[i])
        

df_cleaned.to_csv('filled_missing_val_data.csv', index=False)
#print(df_cleaned)

#part a
mean_list =[0,0]  # null for mean of dates and station id
for i in range(2,len(indep_vari)):
    col = df_cleaned[indep_vari[i]]
    n = len(col)
    sum1 = 0
    for j in col:
        sum1 += j
    mean = sum1/n
    mean_list.append(mean)
    #min and max
    minimum = col[0]
    maximum = col [0]
    for k in col:
        if k < minimum:
            minimum = k
        elif k > maximum:
            maximum = k
    #median
    if n%2 == 0:
        median = (col[n/2] + col[(n/2)+1])/2
    else :   
        median = col[(n+1)/2]
    #Std dev
    sq_diff_sum = 0
    for l in col:
        sq_diff_sum += (l - mean) ** 2

    std_deviation = (sq_diff_sum/(n-1))**(0.5)
    print("The new statistical measures of" ,indep_vari[i], "attribute are: mean=",mean,"maximum=",maximum,"minimum=", minimum,"median=",median,"STD=",std_deviation)
    print()

#original:
get_original_stats()
    
    
