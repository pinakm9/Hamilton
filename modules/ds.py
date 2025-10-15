import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML, display
import utility as ut

class DynamicalSystem:

    def __init__(self, eqnRHS, eqnParams, constraint=None):
        self.eqnRHS = eqnRHS
        self.eqnParams = eqnParams
        self.constraint = constraint  


    
    @ut.timer
    def solve(self, x0, tEnd=20, numEvals=2000):
        timeDerivative = lambda t, x: self.eqnRHS(t, x, self.eqnParams)
        tEvals = np.linspace(0.0, tEnd, numEvals)
        # Enable dense_output to get a continuous solution .sol
        self.soln = solve_ivp(
            timeDerivative, (0.0, tEnd), x0,
            method="LSODA", rtol=1e-8, atol=1e-10, max_step=1e-2,
            dense_output=True
        )

        # Sample the continuous solution at tEvals
        Y = self.soln.sol(tEvals)      # shape (n_states, numEvals)

        # Overwrite t and y so downstream code sees the sampled grid
        self.soln.t = tEvals
        self.soln.y = Y
        

    def getContraintArgs(self):
        q = self.soln.y[:2].T
        p = self.soln.y[2:].T
        return q, p
    

    def plotSolution(self, numBins=100, constraintLabel=None):
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))

        # Phase plot
        axs[0].plot(self.soln.y[0, :], self.soln.y[1, :])
        axs[0].set_xlabel(r'$q_0$')
        axs[0].set_ylabel(r'$q_1$')

        # Compute constraint values
        self.constraintValues = self.constraint(*self.getContraintArgs())

        # Histogram
        axs[1].hist(self.constraintValues, bins=numBins, density=True, alpha=0.7, color='steelblue')
        axs[1].set_title('Constraint Value' if constraintLabel is None else constraintLabel)
        axs[1].set_xlabel('Value')
        axs[1].set_ylabel('Density')

        # Add vertical line for the mean
        mean_val = np.mean(self.constraintValues)
        axs[1].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.4f}')
        axs[1].legend()

        plt.tight_layout()
        plt.show()
        return self.constraintValues





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