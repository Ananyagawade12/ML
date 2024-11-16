import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
data = pd.read_csv('Iris.csv')
matrix = data.drop(columns=['Species'])
y = data['Species'].values
matrix_nparr = np.array(matrix)
y = np.array(y)

print("Attributes (matrix):")
#print(matrix_nparr)         #uncomment later
print("True class labels (y):")
#print(y)                  uncomment later

#outlier replacement part 2
indep_vari = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
out_list = []
total_columns = data.shape[1]       # .shape gives (no. of rows,no. of columns)
for i in range(len(indep_vari)):
    col_name = indep_vari[i]
    sorted_df = matrix.sort_values(by=col_name)
    n = len(sorted_df[col_name])
    if n % 2 == 1:
        median = sorted_df[col_name][n // 2]
    else:
        median = (sorted_df[col_name][n // 2 - 1] + sorted_df[col_name][n // 2]) / 2

    q1 = sorted_df[col_name].quantile(0.25)              #25th percentile
    q3 = sorted_df[col_name].quantile(0.75)              #75th percentile
    IQR = q3 - q1

    for j in range(n) :
        attribute_value = sorted_df[col_name][j]
        if not (q1 - (1.5*IQR)) < attribute_value < (q3 + (1.5*IQR)): #hence outlier
            out_list.append(attribute_value)
            matrix.at[j,col_name] = median
    #print("outliers",col_name," ",out_list)
matrix.to_csv("outlier_corrected.csv",index=False)
X = matrix 
#print("X",X)
print()
copy_X = X.copy()
for i in range(len(indep_vari)):
    col_name = indep_vari[i]
    col_mean = X[col_name].mean()
    #print(col_mean,col_name)
    total_rows = len(X[col_name])
    for j in range(total_rows):
        X.at[j,col_name] = X[col_name][j]-col_mean    #X-
#print("x-u",X)
X_transpose = X.T
#print(X_transpose)
C = X_transpose @ X
#print(C)
#3
eigenvalues, eigenvectors = np.linalg.eigh(C)
#print(eigenvalues)
sorted_indices = np.argsort(eigenvalues)[::-1]  
top_eigenvalues = eigenvalues[sorted_indices[:2]]
top_eigenvectors = eigenvectors[ :,sorted_indices[:2]]
Q=top_eigenvectors
#print(Q)
Q_transpose = Q.T
print("q transpose")
print(Q_transpose)
projected_data = np.matmul(X,Q)   #X^
print(projected_data)  #uncomment

#part d
plt.scatter(projected_data[0], projected_data[1], label='Reduced Data', color='blue')

#plt.show()
plt.quiver(
    0,0,
    top_eigenvectors[0, 0], top_eigenvectors[0, 1],
    angles='xy', scale_units='xy', scale=1, color='r', label='Eigenvector 1'
)
plt.quiver(
    0,0,
    top_eigenvectors[1, 0], top_eigenvectors[1, 1],
    angles='xy', scale_units='xy', scale=1, color='g', label='Eigenvector 2'
)
plt.legend()
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('2D Data with Eigen Directions')
plt.grid(True)
plt.legend()
#plt.xticks(range(-1, 4, 1))
#plt.yticks(range(-8, 3, 1))
plt.show()  #uncomment

#part e
reconstructed_matrix = projected_data @ Q_transpose
reconstructed_matrix.columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
print(reconstructed_matrix)
#part f Rmse b/w original and reconstructed_matrix
rmse_list = [0,0,0,0]
for k in range(len(indep_vari)):
    var = indep_vari[k]
    for i in range(total_rows):
        rmse_list[k] += ((copy_X[var][i] - reconstructed_matrix[var][i])**2)

#print(rmse_list)
rmse_vals = [(i/total_rows)**0.5 for i in rmse_list]
#print(rmse_vals)
print("Rmse : ",rmse_vals)
plt.plot(indep_vari,rmse_vals)
plt.xlabel("attributes")
plt.ylabel("RMSE values")
plt.yticks(range(0,6,1))
plt.show()  #uncomment

#part II  KNN
X_train, X_test, y_train, y_test = train_test_split(projected_data, y, random_state=104, test_size=0.20, shuffle=True)
#print(X_train, X_test, y_train, y_test)
#print("Xtest",X_test)
K = 5
# Function to calculate Euclidean distance between two points
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Lists to store the actual and predicted labels
actual_labels = []
predicted_labels = []

# Loop through each test sample
for i in range(len(X_test)):
    actual_label = y_test[i]
    test_sample = X_test.iloc[i]
    #print(test_sample)
    distances = []  # List to store distances for each test sample
    
    # Compute distance from the test sample to each training sample
    for j in range(len(X_train)):
        train_sample = X_train.iloc[i]
        train_label = y_train[i]
        distance = euclidean_distance(test_sample, train_sample)
        distances.append((distance, train_label)) 

    distances.sort(key=lambda x: x[0])
    k_nearest_neighbors = distances[:5]
    neighbor_labels = [label for _, label in k_nearest_neighbors]
    most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
    actual_labels.append(actual_label)
    predicted_labels.append(most_common_label)

results_matrix = np.column_stack((actual_labels, predicted_labels))
print("Results matrix (actual vs predicted):")
print(results_matrix)

conf_matrix = confusion_matrix(y_test, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.unique(y))
disp.plot(cmap=plt.cm.Greens)
plt.title("Confusion Matrix for KNN Classifier")
plt.show()