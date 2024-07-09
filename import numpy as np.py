import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score



#Load the Iris dateset
iris = load_iris()
x = iris.data
y = iris.target

print("Shape of x:", x.shape)
print("Shape of y:", y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

sample = np.array([5.1, 3.5, 1.4, 0.2]).reshape(1, -1)
print("predicted class:", iris.target_names[model.predict(sample)[0]])


