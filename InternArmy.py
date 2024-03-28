import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from xgboost import XGBClassifier

X = df.drop(columns=['phishing'], axis=1)
y = df['phishing']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)


xgb_clf = XGBClassifier(use_label_encoder=False)
xgb_clf.fit(X_train, y_train)

preds = xgb_clf.predict(X_train)

print(accuracy_score(y_train, preds))
print(recall_score(y_train, preds))
print(precision_score(y_train, preds))

test_preds = xgb_clf.predict(X_test)

print(accuracy_score(y_test, test_preds))
print(recall_score(y_test, test_preds))
print(precision_score(y_test, test_preds))

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, test_preds)