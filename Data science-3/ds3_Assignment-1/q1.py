# Q1
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('landslide_data_original.csv')
#dkdk
indep_vari = ["dates","stationid","temperature","humidity","pressure","rain","lightavg","lightmax","moisture"]
mean_list =[0,0]  # null for man of dates and station id
print_list = []
for i in range(2,len(indep_vari)):
    col = df[indep_vari[i]]
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
    sorted_col = sorted(col)
    if n%2 == 0:
        median = (sorted_col[n // 2 - 1] + sorted_col[n // 2 ])/2
    else :   
        median = sorted_col[n // 2]
    #Std dev
    sq_diff_sum = 0
    for l in col:
        sq_diff_sum += (l - mean) ** 2

    std_deviation = (sq_diff_sum/(n-1))**(0.5)
    print_list.append(f"The statistical measures of original data of {indep_vari[i]} attribute are: mean= {round(mean,2)} maximum= {round(maximum,2)} minimum= {round(minimum,2) } median= {round(median,2)} STD= {round(std_deviation,2)}")
    print("The statistical measures of" ,indep_vari[i], "attribute are: mean=",round(mean,2),"maximum=",round(maximum,2),"minimum=",round(minimum,2) ,"median=",round(median,2),"STD=",round(std_deviation,2))
    print()
#print("maean list",mean_list)
def get_original_stats ():
     for element in print_list:
        print( element)
        print()
        exit 
#pearson correlation
def pearson_correlation(x,y):  # x,y int
        colx = df[indep_vari[x]]
        coly = df[indep_vari[y]]
        mean_x = mean_list[x]
        mean_y = mean_list[y]
        num = 0
        sum_squared_dev_x = 0
        sum_squared_dev_y = 0
        for i in range(len(colx)):
            dev_x = colx[i] - mean_x
            dev_y = coly[i] - mean_y
            num += dev_x * dev_y
            sum_squared_dev_x += dev_x ** 2
            sum_squared_dev_y += dev_y ** 2
        deno = (sum_squared_dev_x ** 0.5) * (sum_squared_dev_y ** 0.5)
        pearson_corr = num / deno
        return pearson_corr
corr_matrix = []
for p1 in range (2, len(indep_vari)):    #p1,p2 are int
    l1 = []
    for p2 in range (2, len(indep_vari)):
            p_corr = pearson_correlation(p1,p2)
            rounded_val = round(p_corr, 4)
            l1.append(rounded_val)
    corr_matrix.append(l1)
headings = ["temperature","humidity","pressure","rain","lightavg","lightmax","moisture"]
pearson_corr_matrix_df = pd.DataFrame(corr_matrix, columns=headings,index=headings)
print(pearson_corr_matrix_df)   
for i in range(len(pearson_corr_matrix_df["lightavg"])): 
     if (pearson_corr_matrix_df["lightavg"][i] <= -0.6 or pearson_corr_matrix_df["lightavg"][i] >= 0.6) and pearson_corr_matrix_df["lightavg"][i] !=1 :
          print("Attribute that is highly correlated with lightavg is",headings[i])   

#histogram
humidity_list_t12 = []
for int_i in range(len(df["stationid"])):
     if df["stationid"][int_i] == "t12":
          humidity_list_t12.append(df["humidity"][int_i])

bin_size = 5
xlist = np.arange(min(humidity_list_t12), max(humidity_list_t12) + bin_size, bin_size)

freq = [0 for i in range(0,len(xlist)-1)]
for val in humidity_list_t12:
     for i in range(len(xlist)):
          if val >= xlist[i] and val < xlist[i+1]:
               freq[i] += 1
               break

plt.bar(xlist[:-1], freq, width=bin_size, edgecolor='black', align='edge')
plt.title('Histogram of Humidity for Station t12')
plt.xlabel('Humidity')
plt.ylabel('Frequency')
plt.xticks(xlist)
plt.show()
   