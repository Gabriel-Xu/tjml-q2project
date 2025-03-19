# No optimization
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import time

df=pd.read_csv("/content/drive/MyDrive/12th Grade 2024-2025/ML/Quarter 2 Project/breast-cancer.csv")
X=df.iloc[:, :-1]
y=df.iloc[:, -1]
encoder=OneHotEncoder()
X_encoded=encoder.fit_transform(X)
X_train, X_test, y_train, y_test=train_test_split(X_encoded, y, test_size=0.2, random_state=0)
start=time.time()
model=RandomForestClassifier(random_state=0, n_jobs=-1)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
end=time.time()
print("Time:", end-start)
accuracy=accuracy_score(y_test, y_pred)
precision=precision_score(y_test, y_pred, pos_label='recurrence-events')
recall=recall_score(y_test, y_pred, pos_label='recurrence-events')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
