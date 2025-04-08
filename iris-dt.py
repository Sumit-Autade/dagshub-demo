import mlflow 
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score , confusion_matrix       
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#load the dataset 
iris = load_iris()
X = iris.data
y = iris.target

# split the dataset into training and testing data 
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 42) 

# define the parameters 

max_depth = 10

# apply mlflow 

mlflow.set_experiment('iris-dt')
with mlflow.start_run():

    dt = DecisionTreeClassifier( max_depth = max_depth)

    dt.fit(X_train , y_train)

    y_pred = dt.predict(X_test)

    accuracy = accuracy_score(y_test , y_pred)

    mlflow.log_metric("accuracy" , accuracy)

    
    mlflow.log_param("max_depth" , max_depth)

    cm = confusion_matrix(y_test , y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')

    mlflow.log_artifact(__file__)

    mlflow.sklearn.log_model(dt, "model")

    mlflow.set_tag("author" , "sumit")

    print("accuracy" , accuracy)