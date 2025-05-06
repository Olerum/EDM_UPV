import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datacleaner import autoclean

# ---- Load data -----------------------------------------------------------
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

clean_train = autoclean(train)

X_train = clean_train.drop("SalePrice", axis=1)
Y_train = clean_train["SalePrice"]

print(X_train.head())
print(Y_train.head())

# ---- Train data -----------------------------------------------------------
model = autosklearn.classification.AutoSklearnClassifier()
model.fit(X_train, Y_train)


# ---- Predict data -----------------------------------------------------------
predictions = model.predict(X_train)

submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': predictions
})
submission.to_csv('/kaggle/working/first.csv', index=False)
print("Submission file has been created successfully.")

# ---- Upload data -----------------------------------------------------------

# Write this in terminal 
#kaggle competitions submit titanic -f submission_autoskl.csv -m "submission_autoskl"