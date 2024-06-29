# -*- coding: utf-8 -*-
"""
Stock Price Prediction & Forecasting with LSTM Neural Networks in Python
Credit to: https://www.youtube.com/watch?v=CbTU92pbDKw&ab_channel=GregHogg

@author: Haby
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

class StockPricePredictor:
    def __init__(self, path, window, p1=0.8, p2=0.9):
        self.path = path
        self.window = window
        self.p1 = p1
        self.p2 = p2

    def load_data(self):
        df = pd.read_csv(self.path)
        df = df[['Date', 'Close']]
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    def plot_data(self, df):
        plt.plot(df.Date, df.Close)
        plt.show()

    def windowed_df(self, data):
        return_df = pd.DataFrame(data['Date'].values, columns=['Date'])
        for i in range(self.window):
            column_header = f'target - {i + 1}'
            column = pd.DataFrame(data['Close'].iloc[i:].values)
            return_df[column_header] = column
        return_df['target'] = pd.Series(data['Close'].iloc[self.window:].values)
        return_df = return_df.dropna()
        return return_df['Date'], return_df['target'], return_df.iloc[:, 1:-1]

    def transform_data(self, x, y):
        scaler = MinMaxScaler()
        y = scaler.fit_transform(np.array(y).reshape(-1, 1))
        x = scaler.fit_transform(x)
        y = torch.from_numpy(y).float()
        x = torch.from_numpy(x).float()
        return x, y, scaler

    def split_data(self, date, x, y):
        tra_l = int(len(date) * self.p1)
        val_l = int(len(date) * self.p2)

        date_train, x_train, y_train = date[:tra_l], x[:tra_l], y[:tra_l]
        date_val, x_val, y_val = date[tra_l:val_l], x[tra_l:val_l], y[tra_l:val_l]
        date_test, x_test, y_test = date[val_l:], x[val_l:], y[val_l:]
        return date_train, x_train, y_train, date_val, x_val, y_val, date_test, x_test, y_test

    def train_model(self, model, x_train, y_train, x_test=None, y_test=None):
        loss_fn = nn.MSELoss(reduction='sum')
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        num_epochs = 30

        train_hist = np.zeros(num_epochs)
        test_hist = np.zeros(num_epochs)

        for t in range(num_epochs):
            model.initial_weight()
            y_pred = model(x_train)
            loss = loss_fn(y_pred, y_train)

            if x_test is not None:
                with torch.no_grad():
                    y_test_pred = model(x_test)
                    test_loss = loss_fn(y_test_pred, y_test)
                test_hist[t] = test_loss.item()
                if t % 5 == 0:
                    print(f'Epoch {t} train loss: {loss.item()} | test loss: {test_loss.item()}')
                    print('-' * 40)
            elif t % 5 == 0:
                print(f'Epoch {t} train loss: {loss.item()}')

            train_hist[t] = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model.eval(), train_hist, test_hist

    def predict(self, model, x_test):
        with torch.no_grad():
            test_seq = x_test[:1]
            preds = []
            for _ in range(len(x_test)):
                y_test_pred = model(test_seq)
                pred = torch.flatten(y_test_pred).item()
                preds.append(pred)
                new_seq = test_seq.numpy().flatten()
                new_seq = np.append(new_seq, [pred])
                new_seq = new_seq[1:]
                test_seq = torch.as_tensor(new_seq).view(1, self.window, 1).float()
        return preds

    def plot_results(self, train_data, true_cases, predicted_cases, daily_cases):
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
        plt.legend()
        plt.show()


if __name__ == "__main__":
    path = r"./Data/MSFT.csv"
    window = 3
    predictor = StockPricePredictor(path, window)

    df = predictor.load_data()
    predictor.plot_data(df)
    date, y, x = predictor.windowed_df(df)
    x, y, scaler = predictor.transform_data(x, y)
    date_train, x_train, y_train, date_val, x_val, y_val, date_test, x_test, y_test = predictor.split_data(date, x, y)

    # Define and train the model
    model = stock_pred_lstm(input_dim=1, hidden_dim=128, seq_length=window, num_layer=2)
    model, train_hist, test_hist = predictor.train_model(model, x_train, y_train, x_test, y_test)

    # Plot training and testing loss
    plt.plot(train_hist, label="train_loss")
    plt.plot(test_hist, label="test_loss")
    plt.legend()
    plt.show()

    # Make predictions
    preds = predictor.predict(model, x_test)

    # Reverse scaling for true and predicted cases
    true_cases = scaler.inverse_transform(np.expand_dims(y_test.flatten().numpy(), axis=0)).flatten()
    predicted_cases = scaler.inverse_transform(np.expand_dims(preds, axis=0)).flatten()

    # Plot results
    daily_cases = df.set_index('Date')
    train_data = x_train.numpy()
    predictor.plot_results(train_data, true_cases, predicted_cases, daily_cases)
