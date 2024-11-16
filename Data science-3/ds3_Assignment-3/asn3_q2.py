import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

train_data = pd.read_csv('iris_train.csv')
test_data = pd.read_csv('iris_test.csv')

def meanCalc(arr):
    return sum(arr) / len(arr)

def covCalc(X):
    n = X.shape[0]
    mn = np.mean(X, axis=0)
    data_meaned = X - mn
    return (1 / (n - 1)) * np.dot(data_meaned.T, data_meaned)

XTrain = train_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
YTrain = train_data['Species'].values
XTest = test_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
YTest = test_data['Species'].values
#print(XTrain)
#classes
classes = np.unique(YTrain)

#params
params = {}
for class_label in classes:
    classData = XTrain[YTrain == class_label]
    mn = np.mean(classData, axis=0)
    cov_matrix = covCalc(classData) 
    prior = len(classData) / len(XTrain)
    params[class_label] = {
        "mean": mn,
        "covariance": cov_matrix,
        "prior": prior
    }

#bays multivariate classification
class_predictions = []
for x in XTest:
    posterior = {}
    for class_label in classes:
        likelihood = multivariate_normal.pdf(x, mean=params[class_label]["mean"], cov=params[class_label]["covariance"])
        posterior[class_label] = likelihood * params[class_label]["prior"]
    
    class_predictions.append(max(posterior, key=posterior.get))

#confusion matrix 
confusion_mat = confusion_matrix(YTest, class_predictions, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

acc= 0
for i in range(len(confusion_mat)):
    acc += confusion_mat[i][i]
accuracy = acc/len(YTest)* 100
print(f"Accuracy: {accuracy:.2f}%")

