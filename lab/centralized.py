import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.nn import functional as F


class HeartDiseaseNN(nn.Module):
    def __init__(self):
        super(HeartDiseaseNN, self).__init__()
        self.fc1 = nn.Linear(30, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 2)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.dropout(x)
        return self.fc4(x)


if __name__ == "__main__":
    df = pd.read_csv("heart.csv")
    categorical = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    encoded_df = pd.get_dummies(df, columns=categorical)
    X = encoded_df.drop("target", axis=1)
    y = encoded_df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()
    y_train = torch.tensor(y_train.values).long()
    y_test = torch.tensor(y_test.values).long()
    net = HeartDiseaseNN()
    optimizer = optim.AdamW(net.parameters())
    criterion = nn.CrossEntropyLoss()

    losses = []
    max_test = 0
    best_params = net.state_dict()
    for epoch in range(1, 50):
        optimizer.zero_grad()
        outputs = net(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        _, preds_y = torch.max(outputs, 1)
        train_acc = accuracy_score(y_train, preds_y)

        pred_test = net(X_test)
        _, preds_test_y = torch.max(pred_test, 1)
        test_acc = accuracy_score(y_test, preds_test_y)
        print("Epoch {}, Loss: {}, Acc:{:.2f}%, Test Acc: {:.2f}%".format(epoch, loss.item(),
                                                                          train_acc * 100, test_acc * 100))
        if test_acc > max_test:
            max_test = test_acc
            best_params = net.state_dict()
    net.load_state_dict(best_params)

