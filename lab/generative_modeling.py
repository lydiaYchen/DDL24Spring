from torch import nn, optim
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from centralized import HeartDiseaseNN as EvaluatorModel


class Autoencoder(nn.Module):
    def __init__(self, D_in, H=50, H2=12, latent_dim=3):

        # Encoder
        super(Autoencoder, self).__init__()
        self.optimizer = None
        self.criterion = None
        self.linear1 = nn.Linear(D_in, H)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)
        self.linear2 = nn.Linear(H, H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3 = nn.Linear(H2, H2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H2)

        #         # Latent vectors mu and sigma
        self.fc1 = nn.Linear(H2, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

        #         # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H2)
        self.fc_bn4 = nn.BatchNorm1d(H2)

        #         # Decoder
        self.linear4 = nn.Linear(H2, H2)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H2)
        self.linear5 = nn.Linear(H2, H)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H)
        self.linear6 = nn.Linear(H, D_in)
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)

        self.relu = nn.ReLU()

    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)

        return r1, r2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        return self.lin_bn6(self.linear6(lin5))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def train_with_settings(self, epochs, batch_sz, real_data, optimizer, loss_fn):
        self.optimizer = optimizer
        self.criterion = loss_fn
        num_batches = len(real_data) // batch_sz if len(real_data) % batch_sz == 0 else len(real_data) // batch_sz + 1
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            total_loss = 0.0
            for minibatch in range(num_batches):
                if minibatch == num_batches - 1:
                    minibatch_data = real_data[int(minibatch * batch_sz):]
                else:
                    minibatch_data = real_data[int(minibatch * batch_sz):int((minibatch + 1) * batch_sz)]

                outs, mu, logvar = self.forward(minibatch_data)
                loss = self.criterion(outs, minibatch_data, mu, logvar)
                total_loss += loss
                loss.backward()
                self.optimizer.step()

            print(
                f"Epoch: {epoch} Loss: {total_loss.detach().numpy() / num_batches:.3f}")

    def sample(self, nr_samples, mu, logvar):
        sigma = torch.exp(logvar / 2)
        no_samples = nr_samples
        q = torch.distributions.Normal(mu.mean(axis=0), sigma.mean(axis=0))
        z = q.rsample(sample_shape=torch.Size([no_samples]))
        with torch.no_grad():
            pred = self.decode(z).cpu().numpy()

        pred[:, -1] = np.clip(pred[:, -1], 0, 1)
        pred[:, -1] = np.round(pred[:, -1])
        return pred


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    df = pd.read_csv(Path(__file__).parent / "heart-dataset" / "heart.csv")
    categorical = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    encoded_df = pd.get_dummies(df, columns=categorical)
    X = encoded_df.drop("target", axis=1)
    y = encoded_df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()
    y_train = torch.tensor(y_train.values).long()
    y_test = torch.tensor(y_test.values).long()
    D_in = X.shape[1] + 1
    H = 48
    H2 = 32
    latent_dim = 16
    model = Autoencoder(D_in, H, H2, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_mse = customLoss()
    real_data = torch.concat((X_train, y_train.view(-1, 1)), dim=1)
    EPOCHS = 200
    BATCH_SIZE = 64
    model.train_with_settings(EPOCHS, BATCH_SIZE, real_data, optimizer, loss_mse)

    _, mu, logvar = model.forward(real_data)

    synthetic_data = model.sample(len(real_data), mu, logvar)
    synthetic_x = torch.tensor(synthetic_data[:, :-1])
    synthetic_y = torch.tensor(synthetic_data[:, -1]).long()

    print("--------------Testing model trained on real data----------")
    evalm1 = EvaluatorModel()
    opt1 = optim.AdamW(evalm1.parameters())
    criterion = nn.CrossEntropyLoss()

    losses = []

    for epoch in range(1, 50):
        opt1.zero_grad()
        outputs = evalm1(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        opt1.step()
        losses.append(loss.item())
        _, preds_y = torch.max(outputs, 1)
        train_acc = accuracy_score(y_train, preds_y)

        pred_test = evalm1(X_test)
        _, preds_test_y = torch.max(pred_test, 1)
        test_acc = accuracy_score(y_test, preds_test_y)
        print("Epoch {}, Loss: {:.2f}, Acc:{:.2f}%, Test Acc: {:.2f}%".format(epoch, loss.item(),
                                                                              train_acc * 100, test_acc * 100))

    print("--------------Testing model trained on synthetic data----------")
    evalm2 = EvaluatorModel()
    opt2 = optim.AdamW(evalm2.parameters())
    criterion = nn.CrossEntropyLoss()

    losses = []

    for epoch in range(1, 50):
        opt2.zero_grad()
        outputs = evalm2(synthetic_x)
        loss = criterion(outputs, synthetic_y)
        loss.backward()
        opt2.step()
        losses.append(loss.item())
        _, preds_y = torch.max(outputs, 1)
        train_acc = accuracy_score(synthetic_y, preds_y)

        pred_test = evalm2(X_test)
        _, preds_test_y = torch.max(pred_test, 1)
        test_acc = accuracy_score(y_test, preds_test_y)
        print("Epoch {}, Loss: {:.2f}, Acc:{:.2f}%, Test Acc: {:.2f}%".format(epoch, loss.item(),
                                                                              train_acc * 100, test_acc * 100))
