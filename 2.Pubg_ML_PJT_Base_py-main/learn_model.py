from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0xC0FFEE)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0xC0FFEE)


def linear_reg(df):
    X_train, X_val, y_train, y_val = model_train_data(df)
    reg = LinearRegression().fit(X_train, y_train)
    model_pred_eval_test_data(reg,X_val,y_val)
#     return reg
    

def model_train_data(df):
    X = df.drop(columns='winPlacePerc')
    y = df.winPlacePerc
    return train_test_split(X, y, test_size=0.2, random_state=0xC0FFEE)

def model_pred_eval_train_data(model,X,y):
    pred_train = model.predict(X)
    print("train:",mean_absolute_error(y_train, pred_train))


def model_pred_eval_test_data(model,X_val,y_val):
    pred_val = model.predict(X_val)
    print("train:",mean_absolute_error(y_val, pred_val))

