import torch
import torch.nn as nn
from typing import Tuple, List
import utility as ut
from tqdm import tqdm
import numpy as np 
import matplotlib.pyplot as plt

class LSTM(nn.Module):
    """
    Stateless multilayer LSTMCell network implementing R^D -> R^D.
    Each call zero-initializes hidden/cell states, so it's a pure function of x.
    """
    def __init__(self, D: int, hidden_size: int = 64, num_layers: int = 3):
        super().__init__()
        assert D > 0, "D must be positive."
        assert num_layers >= 1, "num_layers must be >= 1."
        self.D = D
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Stack of LSTMCells
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_size = D if i == 0 else hidden_size
            self.layers.append(nn.LSTMCell(in_size, hidden_size))

        # Final projection back to R^D
        self.head = nn.Linear(hidden_size, D)

    def _zero_state(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        batch = x.size(0)
        device, dtype = x.device, x.dtype
        return [
            (
                torch.zeros(batch, self.hidden_size, device=device, dtype=dtype),
                torch.zeros(batch, self.hidden_size, device=device, dtype=dtype),
            )
            for _ in range(self.num_layers)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, D)
        returns: (batch, D)
        """
        assert x.dim() == 2 and x.size(1) == self.D, f"Expected (batch, {self.D})"
        states = self._zero_state(x)

        h_in = x
        for i, layer in enumerate(self.layers):
            h_i, c_i = states[i]
            h_out, c_out = layer(h_in, (h_i, c_i))
            h_in = h_out

        y = self.head(h_out)
        return y




class Surrogate:

    def __init__(self, nnHyperparams, device, modelPath=None):
        self.model = LSTM(*nnHyperparams)
        if modelPath is not None:
            self.load(modelPath)
        self.to(device)

    def __call__(self, x):
        return self.model(x) 
    
    def save(self, path):
        torch.save(self.model.state_dict(), path) 

    def load(self, path):
        self.model.load_state_dict(torch.load(path)) 

    def to(self, device):
        self.device = device
        self.model.to(device) 

    @torch.no_grad()
    def predict(self, x, numSteps):
        y = x.to(self.device)
        trajectory = torch.zeros((numSteps + 1, self.model.D), device=self.device)
        trajectory[0] = y
        for i in range(numSteps):
            y = self.model(y.unsqueeze(0)).squeeze(0)
            trajectory[i + 1] = y
        return trajectory


    @ut.timer
    def train(self, trainData, batchSize=128, learningRate=1e-3, numEpochs=10):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learningRate)

        # Prepare dataset and loader
        self.trainData = trainData
        self.trainData = torch.tensor(self.trainData, device=self.device)
        dataset = torch.utils.data.TensorDataset(self.trainData[:-1], self.trainData[1:])
        trainLoader = torch.utils.data.DataLoader(
            dataset, batch_size=batchSize, shuffle=True, drop_last=False
        )

        for epoch in range(numEpochs):
            epoch_loss = 0.0

            # tqdm progress bar over batches
            pbar = tqdm(trainLoader, desc=f"Epoch {epoch+1}/{numEpochs}", leave=False, ncols=100)
            for x, y in pbar:
                optimizer.zero_grad()
                y_hat = self.model(x)
                loss = torch.sum((y_hat - y) ** 2, dim=1).mean()
                loss.backward()
                optimizer.step()

                # Update running loss
                epoch_loss += loss.item() * x.size(0)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})  # live update

            epoch_loss /= len(dataset)
            tqdm.write(f"Epoch {epoch+1}/{numEpochs} | Mean Loss: {epoch_loss:.6f}", end="")



    def plotExperiment(self, numSteps=1000):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the solution
        predictedTrajectory = self.plotSolution(ax[0], numSteps)
        return predictedTrajectory


    @ut.timer
    def plotSolution(self, ax, numSteps):
        
        # Plot the training data
        trainData = self.trainData.cpu().numpy()
        ax.scatter(trainData[:, 0], trainData[:, 1], color='blue', label='Training Data', alpha=0.1, s=0.1)

        # Generate an plot trajectory
        predictedTrajectory = self.predict(self.trainData[0], numSteps).cpu().numpy()
        ax.plot(predictedTrajectory[:, 0], predictedTrajectory[:, 1], linestyle='-', color='black', label='Predicted Trajectory') 

        # Labels and title
        ax.set_xlabel(r'$q_0$')
        ax.set_ylabel(r'$q_1$')
        ax.legend()

        return predictedTrajectory
        
        

        