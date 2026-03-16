import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("C:/Users/DELL/OneDrive/Desktop/python course(asra)/diabetes.csv")
print(df.head())
#EDA
print(df.shape)#rows and columns
df.columns #column names
df.describe()#mean,mini,max,std
df.isnull().sum()#null rows and columns
#correlation heatmap
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))#for set plot size
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()
#target variable distribution
df["Outcome"].value_counts()
x = df.drop("Outcome", axis=1)
y = df["Outcome"]
#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
#model training
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("starting model training...")
#model create
model = LogisticRegression(max_iter=1000)
#train model
model.fit(x_train, y_train)
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)
#predict on test data
y_predict = model.predict(x_test)
print("prdiction done....")
print("accuracy score:", accuracy_score(y_test, y_predict))
print("classification report:\n", classification_report(y_test, y_predict))
print("confusion matrix:\n", confusion_matrix(y_test, y_predict))
import matplotlib.pyplot as plt
from sklearn.metrics import  ConfusionMatrixDisplay
#confusiob matrix plot
ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, cmap="Blues",display_labels=["non-diabetic", "diabetic"])
plt.show()
import joblib
#load model
loaded_model = joblib.load("diabetes_model.pkl")
#prdict new data
new_data = [[2, 120, 70, 30, 100, 25.5, 0.5, 33]]#example data
prediction = loaded_model.predict(new_data)
print("prediction for new data:", "diabetic" if prediction[0] == 1 else "non-diabetic")


