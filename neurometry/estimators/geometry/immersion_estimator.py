import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs
import torch
import torch.optim as optim
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import neurometry.estimators.geometry.models.train_config as train_config


class ImmersionEstimator(BaseEstimator):
    def __init__(self, extrinsic_dim, topology, device, verbose=False):
        self.estimate_ = None
        self.verbose = verbose
        self.device = device
        self.extrinsic_dim = extrinsic_dim
        self.topology = topology
        self.latent_dims = {"circle": 2, "sphere": 3, "torus": 4}
        self.model = self._get_model()

    def _get_model(self):
        return NeuralEmbedding(
            latent_dim=self.latent_dims[self.topology], extrinsic_dim=self.extrinsic_dim
        ).to(self.device)

    def intrinsic_to_extrinsic(self, x):
        if self.topology == "circle":
            return gs.array([gs.cos(x), gs.sin(x)]).T
        if self.topology == "sphere":
            return gs.array(
                [
                    gs.sin(x[:, 0]) * gs.cos(x[:, 1]),
                    gs.sin(x[:, 0]) * gs.sin(x[:, 1]),
                    gs.cos(x[:, 0]),
                ]
            ).T
        if self.topology == "torus":
            return gs.array(
                [
                    gs.cos(x[:, 0]),
                    gs.sin(x[:, 0]),
                    gs.cos(x[:, 1]),
                    gs.sin(x[:, 1]),
                ]
            ).T
        raise ValueError("Topology not supported")

    def fit(self, X, y=None):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train = torch.tensor(X_train).to(self.device)
        X_test = torch.tensor(X_test).to(self.device)
        y_train = torch.tensor(y_train).to(self.device)
        y_test = torch.tensor(y_test).to(self.device)
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=train_config.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=train_config.batch_size,
            shuffle=True,
        )

        trainer = Trainer(
            self.model,
            train_loader,
            test_loader,
            criterion=torch.nn.MSELoss(),
            learning_rate=train_config.lr,
            scheduler=False,
            verbose=self.verbose,
        )
        trainer.train(train_config.num_epochs)

        self.trainer = trainer

        self.model.eval()
        self.estimate_ = lambda task_variable: self.model(
            self.intrinsic_to_extrinsic(task_variable)
        )

        return self


class NeuralEmbedding(torch.nn.Module):
    def __init__(
        self, latent_dim, extrinsic_dim, hidden_dims=64, num_hidden=4, sft_beta=4.5
    ):
        super().__init__()

        self.fc1 = torch.nn.Linear(latent_dim, hidden_dims)
        self.fc_hidden = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dims, hidden_dims) for _ in range(num_hidden)]
        )
        self.fc_output = torch.nn.Linear(hidden_dims, extrinsic_dim)
        self.softplus = torch.nn.Softplus(beta=sft_beta)

    def forward(self, x):
        h = self.softplus(self.fc1(x))
        for fc in self.fc_hidden:
            h = self.softplus(fc(h))
        return self.fc_output(h)


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        criterion,
        learning_rate,
        scheduler=False,
        verbose=False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        if scheduler:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )
        self.verbose = verbose

    def train(self, num_epochs=10):
        train_losses = []
        test_losses = []
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            for inputs, targets in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)
            avg_test_loss = self.evaluate()
            test_losses.append(avg_test_loss)
            if self.verbose:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Test Loss: {avg_test_loss}"
                )
        self.train_losses = train_losses
        self.test_losses = test_losses

    def evaluate(self):
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
        return test_loss / len(self.test_loader)
