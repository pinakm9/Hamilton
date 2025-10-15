import torch
import torch.nn as nn
from typing import Tuple, List
import utility as ut
from tqdm import tqdm

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
            y = self.model(y)
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
            pbar = tqdm(trainLoader, desc=f"Epoch {epoch+1}/{numEpochs}", leave=False)
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
            tqdm.write(f"Epoch {epoch+1}/{numEpochs} | Mean Loss: {epoch_loss:.6f}")




    @ut.timer
    def plotSolutionAnimated(self, numBins=100, constraintLabel=None,
                            interval=20, show_html=True, save=None, fps=30,
                            batch_size=50,
                            max_frames=None):
        """
        Animate adding *batch_size* points per frame to:
        (1) q-plane trajectory, (2) normalized histogram with mean line.
        """
        # --- Data
        q = self.soln.y[:2, :]                      # (2, N)
        q1, q2 = np.asarray(q[0]), np.asarray(q[1])
        N = q1.size


        # Constraint values aligned with trajectory
        self.constraintValues = np.asarray(self.constraint(*self.getContraintArgs())).ravel()
        cvals = self.constraintValues
    

        # Precompute histogram bins using full data (stable y-limits)
        hist_full, bin_edges = np.histogram(cvals, bins=numBins, density=True)
        bin_widths = np.diff(bin_edges)
        x_lefts = bin_edges[:-1]
        ymax = max(hist_full.max(), 1e-12)

        # --- Frame schedule (batching)
        ends = np.arange(batch_size, N + batch_size, batch_size)
        ends[-1] = N
        if max_frames is not None and max_frames > 0:
            ends = ends[:max_frames]
            if ends[-1] != N:  # ensure we still show the final frame
                ends = np.append(ends, N)
        frames = ends.tolist()

        # --- Figure/axes
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

        # Left: q-plane
        axs[0].set_xlabel(r"$q_0$"); axs[0].set_ylabel(r"$q_1$")
        axs[0].set_title("Trajectory")
        pad_x = 0.05 * (q1.max() - q1.min() or 1.0)
        pad_y = 0.05 * (q2.max() - q2.min() or 1.0)
        axs[0].set_xlim(q1.min() - pad_x, q1.max() + pad_x)
        axs[0].set_ylim(q2.min() - pad_y, q2.max() + pad_y)
        # axs[0].set_aspect('equal', 'box')

        (line_traj,) = axs[0].plot([], [], lw=2)
        dot_curr = axs[0].scatter([], [], s=36)
        ttext = None#axs[0].text(0.02, 0.98, "", transform=axs[0].transAxes,
        #                     ha='left', va='top')

        # Right: histogram (normalized)
        axs[1].set_title('Constraint Value' if constraintLabel is None else constraintLabel)
        axs[1].set_xlabel("value"); axs[1].set_ylabel("density")
        axs[1].set_xlim(bin_edges[0], bin_edges[-1])
        axs[1].set_ylim(0, ymax * 1.05)

        bars = axs[1].bar(x_lefts, np.zeros_like(x_lefts),
                        width=bin_widths, align='edge', edgecolor='k')

        # --- Mean line & readout (NEW)
        # Place a vertical line for the mean; position is updated each frame.
        mean_line = axs[1].axvline(x=bin_edges[0], linestyle='--', linewidth=2, label='Mean')
        mean_text = axs[1].text(0.98, 0.95, "", transform=axs[1].transAxes,
                                ha='right', va='top')
        axs[1].legend()

        # --- Anim callbacks
        def init():
            line_traj.set_data([], [])
            dot_curr.set_offsets(np.empty((0, 2)))
            for b in bars:
                b.set_height(0.0)
            # ttext.set_text("")
            # Initialize mean display
            mean_line.set_xdata([bin_edges[0], bin_edges[0]])
            mean_text.set_text("")
            return [line_traj, dot_curr, *bars, ttext, mean_line, mean_text]

        def update(i_end):
            # Trajectory
            line_traj.set_data(q1[:i_end], q2[:i_end])
            dot_curr.set_offsets(np.array([[q1[i_end-1], q2[i_end-1]]], dtype=float))

            # Histogram with first i_end samples
            h, _ = np.histogram(cvals[:i_end], bins=bin_edges, density=True)
            for b, h_i in zip(bars, h):
                b.set_height(h_i)

            # Update mean line & text
            m = float(np.mean(cvals[:i_end]))
            mean_line.set_xdata([m, m])
            mean_text.set_text(f"Mean = {m:.4f}")

            return [line_traj, dot_curr, *bars, ttext, mean_line, mean_text]

        ani = FuncAnimation(fig, update, frames=frames, init_func=init,
                            interval=interval, blit=False)

        if save:
            if save.lower().endswith(".gif"):
                ani.save(save, writer=PillowWriter(fps=fps))
            else:
                ani.save(save, fps=fps)  # requires ffmpeg for mp4
            plt.close(fig)
            print(f"Saved animation to {save}")
        elif show_html:
            plt.close(fig)
            display(HTML(ani.to_jshtml()))
        else:
            plt.show()

        self._last_animation = ani
        return self.constraintValues