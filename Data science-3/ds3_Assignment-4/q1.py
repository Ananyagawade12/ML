import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv("abalone.csv")
train_data,test_data = train_test_split(data,test_size = 0.3,random_state = 42)
train_data.to_csv("abalone_train.csv",index = False)
test_data.to_csv("abalone_test.csv",index = False)
means = train_data.mean()
#print(means)
def pearson(train_data,x,y):
    numSum = 0
    denX = 0
    denY = 0
    xi = means[x]
    yi = means[y]
    xList = train_data[x]
    yList = train_data[y]
    for i in range(len(train_data[x])):
        numSum += (xList.iloc[i] - xi)*(yList.iloc[i] - yi)
        denX += (xList.iloc[i] - xi)**2
        denY += (yList.iloc[i] - yi)**2
    den = (denX * denY)**(1/2)
    pearson_coeff = numSum / den

    return pearson_coeff

corr = []
for c in train_data.columns[:-1]:
    corr.append(pearson(train_data,c,"Rings"))

#print(corr)
k = corr.index(max(corr))
c = train_data.columns[k]
#print(k)
def finderror(yn,y):
    s = 0
    n = len(y)
    for i in range(n):
        s += (yn.iloc[i] - y.iloc[i])**2
    s = (s/n)**(1/2)

    return s

 #LINEAR REGRESSION
                                
def findParamlinear(train_data,x,y):
    numSum = 0
    den = 0
    xi = means[x]
    yi = means[y]
    xList = train_data[x]
    yList = train_data[y]
    for i in range(len(train_data[x])):
        numSum += (xList.iloc[i] - xi)*(yList.iloc[i] - yi)
        den += (xList.iloc[i] - xi)**2
    w = numSum / den
    w0 = yi - xi*w

    return w,w0


w,w0 = findParamlinear(train_data,c,"Rings")

chosen = train_data[c]
rings = train_data["Rings"]
test_chosen = test_data[c]
test_rings = test_data["Rings"]

predicted_rings = []
for i in range(len(train_data[c])):
    yn = w*chosen.iloc[i] + w0
    predicted_rings.append(yn)
predicted_rings = pd.Series(predicted_rings)

predicted_test_rings = []
for i in range(len(test_data[c])):
    yn = w*test_chosen.iloc[i] + w0
    predicted_test_rings.append(yn)
predicted_test_rings = pd.Series(predicted_test_rings)

plt.scatter(chosen,rings,color = "green")
plt.xlabel("Chosen attribute")
plt.ylabel("Rings")

plt.plot(chosen,predicted_rings,color = "red")
plt.title("linear regression")
plt.show()
print("accuracy of trained data by linear regression:")
print(finderror(rings,predicted_rings))
print("accuracy of test data by linear regression:")
print(finderror(test_rings,predicted_test_rings))

plt.scatter(test_rings,predicted_test_rings,color = "orange")
plt.xlabel("actual Rings")
plt.ylabel("Predicted Rings")
plt.title("Actual VS Predicted")
plt.show()


#POLYNOMIAL REGRESSION

def findParamPoly(train_data,x,y,p):
    O = np.ones((len(train_data), 1))
    features = train_data[x].to_numpy()
    O = np.hstack([O] + [features.reshape(-1, 1) ** i for i in range(1, p + 1)])
    y = train_data[y].to_numpy()
    w = np.dot(np.linalg.inv(np.dot(O.T,O)),np.dot(O.T,y))

    return w

def predictByP(p):
    w = findParamPoly(train_data,c,"Rings",p)
    predicted_rings = []
    for i in range(len(train_data)):
        x = float(chosen.iloc[i])
        new = [x ** j for j in range(p+1)]
        y = pd.DataFrame({"A":new})
        yn = w@y
        predicted_rings.append(yn)
    predicted_rings = pd.Series(predicted_rings)


    predicted_test_rings = []
    for i in range(len(test_data)):
        x = float(test_chosen.iloc[i])
        new = [x ** j for j in range(p+1)]
        y1 = pd.DataFrame({"A":new})
        yn = w@y1
        predicted_test_rings.append(yn)
    predicted_test_rings = pd.Series(predicted_test_rings)

    return predicted_rings,predicted_test_rings

rmse_train = []
rmse_test = []
for i in range(2,6):
    print()
    print("accuracy of rings in trained data by polynomial regression when p = ",i,":")
    predicted_rings,predicted_test_rings = predictByP(i)
    print((finderror(rings,predicted_rings)).tolist())
    rmse_train.append(((finderror(rings,predicted_rings)).tolist())[0])

    print("accuracy of rings in test data by polynomial regression when p = ",i,":")
    print((finderror(test_rings,predicted_test_rings)).tolist())
    rmse_test.append(((finderror(test_rings,predicted_test_rings)).tolist())[0])
    print()


p = [2,3,4,5]

plt.bar(p, rmse_train)
plt.xlabel('Degree of Polynomial')
plt.ylabel('RMSE')
plt.title('Training RMSE vs Degree of Polynomial')

plt.plot(p,rmse_train,color = "red")

plt.show()

plt.bar(p, rmse_test)
plt.xlabel('Degree of Polynomial')
plt.ylabel('RMSE')
plt.title('Testing RMSE vs Degree of Polynomial')

plt.plot(p,rmse_test,color = "red")
plt.show()
print(rmse_test)

##for i in range(6,10):
##    predicted_rings,predicted_test_rings = predictByP(i)
##    plt.scatter(chosen,predicted_rings)
##    plt.show()


##plt.scatter(chosen,rings,color = "orange")
##
##plt.scatter(test_chosen,test_rings,color = "orange")
##
##plt.scatter(chosen,predicted_rings,color = "blue")
##
##plt.scatter(test_chosen,predicted_test_rings,color = "blue")
##plt.show()
# Instead of using rmse_test.index(min(rmse_test)) + 2
# Create an explicit mapping between degrees and RMSE values
p_values = [2, 3, 4, 5]
minerrorP = p_values[rmse_test.index(min(rmse_test))]

print("minerrorP", minerrorP)  # This should now print the correct degree based on RMSE.

predicted_rings, _ = predictByP(minerrorP)
sorted_indices = np.argsort(chosen)
sorted_chosen = chosen.iloc[sorted_indices]
sorted_predicted_rings = predicted_rings.iloc[sorted_indices]
plt.scatter(chosen, rings, color="green", label="Actual Rings")  
plt.plot(sorted_chosen, sorted_predicted_rings, color="red", label=f"Best-fit Curve (p = {minerrorP})")  
plt.xlabel("Chosen attribute")
plt.ylabel("Rings")
plt.title(f"Best-fit Curve for Polynomial Regression (Degree = {minerrorP})")
plt.legend()
plt.show()

'''

minerrorP = rmse_test.index(min(rmse_test)) + 2  
print("minerrorP",minerrorP)
'''   







    
    
