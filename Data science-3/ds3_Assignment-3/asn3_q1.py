import pandas as pd
from pcaFunction_file import pca
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix

train_data = pd.read_csv('iris_train.csv')
test_data = pd.read_csv('iris_test.csv')
"""
feature_train = train_data.iloc[:, :-1]  
label_train = train_data.iloc[:, -1]  
"""
feature_train = train_data.iloc[:, 1:-1].values    #.values = converts into a NumPy array
label_train = train_data.iloc[:, -1].values       # last column is label column
feature_test = test_data.iloc[:, 1:-1].values
label_test = test_data.iloc[:, -1].values

print(feature_train,feature_test,label_train,label_test)

X_train_pca = pca(feature_train, n_components=1)
X_test_pca = pca(feature_test, n_components=1)
print(X_test_pca,X_train_pca)
def estimate_priors(y_train):
    classes, counts = np.unique(y_train, return_counts=True)    # .unique returns unique vals & no. of occurrences
    priors = {class_label: count / len(y_train) for class_label, count in zip(classes, counts)}
    return priors

def estimate_gaussian_params(X, y, class_label):
    X_class = X[y == class_label]
    mean = np.mean(X_class)
    variance = np.var(X_class)
    return mean, variance

def gaussian_pdf(x, mean, variance):
    coeff = 1.0 / np.sqrt(2 * np.pi * variance)
    exponent = np.exp(-((x - mean) ** 2) / (2 * variance))
    return coeff * exponent

def bayes_classify(x, params, priors):
    posteriors = {}
    for class_label, param in params.items():
        mean = param['mean']
        variance = param['variance']
        likelihood = gaussian_pdf(x, mean, variance)    # P(x | C_i)
        posteriors[class_label] = likelihood * priors[class_label]  #P(C_i | x) = P(x | C_i) * P(C_i)
    return max(posteriors, key=posteriors.get)

classes = np.unique(label_train)
#print(classes)
params = {}
for class_label in classes:
    mean, variance = estimate_gaussian_params(X_train_pca, label_train, class_label)
    params[class_label] = {'mean': mean, 'variance': variance}

priors = estimate_priors(label_train)
y_pred_bayes = [bayes_classify(x, params, priors) for x in X_test_pca]

print(y_pred_bayes)
#part 4
confusion_matrix = confusion_matrix(label_test, y_pred_bayes)
print("Confusion Matrix:")
print(confusion_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
acc= 0
for i in range(len(confusion_matrix)):
    acc += confusion_matrix[i][i]
accuracy = acc/len(label_test)* 100
print(f"Accuracy: {accuracy:.2f}%")
