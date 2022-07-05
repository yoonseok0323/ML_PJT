from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0xC0FFEE)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0xC0FFEE)

def get_test_result(model, test_df):
    pred_test = model.predict(test_df)
    return pred_test


def linear_reg(df):
    X_train, X_val, y_train, y_val = model_train_data(df)
    
    # scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_val = scaler.fit_transform(X_val)
    
    reg = LinearRegression(n_jobs=-1).fit(X_train, y_train)
    model_pred_eval_test_data(reg,X_val,y_val)
#     return reg
    

def model_train_data(df):
    X = df.drop(columns='winPlacePerc')
    y = df.winPlacePerc
    return train_test_split(X, y, test_size=0.2, random_state=0xC0FFEE)

# def model_pred_eval_train_data(model,X,y):
#     pred_train = model.predict(X)
#     print("train:",mean_absolute_error(y_train, pred_train))


def model_pred_eval_test_data(model,X_val,y_val):
    pred_val = model.predict(X_val)
    print("linear_reg:",mean_absolute_error(y_val, pred_val))

def poly_reg(df):
    X_train, X_val, y_train, y_val = model_train_data(df)
    
    # scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_val = scaler.fit_transform(X_val)
    
    poly = PolynomialFeatures(degree=2, include_bias=False,order='F')
    X_train_poly = poly.fit_transform(X_train)    
    reg = LinearRegression()
    reg.fit(X_train_poly, y_train)
    X_val_poly = poly.transform(X_val)
    y_pred = reg.predict(X_val_poly)
    mae_train_p = mean_absolute_error(y_val, y_pred)
    print("poly_reg:",mae_train_p)
    
def graph(pred_val,y):
    fig = plt.figure(figsize =(12,4))
    graph = fig.add_subplot(1,1,1)
    graph.plot(y,marker='o')
    graph.plot(pred_val, marker='^' )
