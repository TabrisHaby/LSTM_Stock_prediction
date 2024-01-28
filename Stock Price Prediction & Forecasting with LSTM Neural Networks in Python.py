# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 19:01:53 2023

Stock Price Prediction & Forecasting with LSTM Neural Networks in Python
Credit to : https://www.youtube.com/watch?v=CbTU92pbDKw&ab_channel=GregHogg

@author: BY
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

path = r"E:\Data_and_Script\Data\MSFT.csv"
df = pd.read_csv(path)
df = df[['Date', 'Close']]
df['Date'] = pd.to_datetime(df["Date"])
# df.index = df.pop('Date')

# plot
plt.plot(df.Date, df.Close)

#%%
# define window function
window = 3
def windowed_df(data,window):
    # develop a window function so that lstm can be applied into TS data
    return_df = pd.DataFrame(data['Date'].values, columns = ['Date'])
    for i in range(window) :
        column_header = 'target - ' + str(i+1)
        column = pd.DataFrame(data['Close'].iloc[i:].values)

        return_df[column_header] = column 
    return_df['target'] = pd.Series(data['Close'].iloc[window:].values)
    return_df = return_df.dropna()
    print(return_df)
    return return_df['Date'], return_df['target'], return_df.iloc[:,1:-1]

date, y, x  = windowed_df(data = df, window = window)
print(date.shape, y.shape, x.shape)
      

#%%
# trasnformation and split
import torch
scaler = MinMaxScaler()
y = scaler.fit_transform(np.array(y).reshape(-1,1)) 
x = scaler.fit_transform(x)

y = torch.from_numpy(y).float()
x = torch.from_numpy(x).float()
# train val and test
p1 = 0.8
p2 = 0.9

tra_l = int(len(date)*p1)
val_l = int(len(date)*p2)

date_train, x_train, y_train = date[:tra_l], x[:tra_l], y[:tra_l]
date_val, x_val, y_val = date[tra_l:val_l], x[tra_l:val_l], y[tra_l:val_l]
date_test, x_test, y_test = date[val_l:], x[val_l:], y[val_l:]
print(date_train.shape, date_val.shape, date_test.shape)

#%%
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.insert(0, r'E:\Data_and_Script\Python_Script')
from model_stock_price import *

def train_model(model, train_data, train_labels, test_data = None, test_labels = None):
    # loss function
    loss_fn = nn.MSELoss(reduction='sum')
    # optimizer Adam
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    
    num_epochs = 30
    
    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)
    
    for t in range(num_epochs):
        
        model.initial_weight()
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        
        if test_data is not None :
            with torch.no_grad():
                y_test_pred = model(x_test)
                test_loss = loss_fn(y_test_pred, y_test)
            test_hist[t] = test_loss.item() 
            
            if t % 5 == 0 :
                print(f'Epoch {t} train loss: {loss.item()} | test loss: {test_loss.item()}')
                print('-'*40)
        elif t % 5 == 0 :
            print(f"Epoch {t} train loss : {loss.item()}")
        
        train_hist[t] = loss.item()
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
    return model.eval(), train_hist, test_hist

#%%
model = stock_pred_lstm(input_dim = 1, hidden_dim=128, seq_length=window, num_layer=2)
model, train_hist, test_hist = train_model(
    model, x_train, y_train, x_test, y_test)

#%%
plt.plot(train_hist, label = "train_loss")
plt.plot(test_hist, label = "test_loss")
#plt.ylim((0, 5))
plt.legend()

#%%
# predict daily case
with torch.no_grad():
    
    test_seq = x_test[:1]
    preds = []
    for i in range(len(x_test)):
        y_test_pred = model(test_seq)
        pred = torch.flatten(y_test_pred).item()
        preds.append(pred)
        
        new_seq = test_seq.numpy().flatten()
        new_seq = np.append(new_seq, [pred])
        new_seq = new_seq[1:]
        test_seq = torch.as_tensor(new_seq).view(1, window, 1).float()

#%%
# reverse the scaling of the test data and the model predictions:
true_cases = scaler.inverse_transform(
    np.expand_dims(y_test.flatten().numpy(), axis=0)
).flatten()

predicted_cases = scaler.inverse_transform(
  np.expand_dims(preds, axis=0)
).flatten()

#%%
plt.plot(
  daily_cases.index[:len(train_data)],
  scaler.inverse_transform(train_data).flatten(),
  label='Historical Stock Price'
)
plt.plot(
  daily_cases.index[len(train_data):len(train_data) + len(true_cases)],
  true_cases,
  label='Real Stock Price'
)
plt.plot(
  daily_cases.index[len(train_data):len(train_data) + len(true_cases)],
  predicted_cases,
  label='Predicted Stock Price'
)
plt.legend();

