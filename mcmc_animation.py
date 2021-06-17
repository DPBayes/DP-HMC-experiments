import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 

class MCMCAnimation: 
    def __init__(self, samples, proposals):
        self.props_per_sample = int(proposals.shape[0] / (samples.shape[0] - 1))
        self.n_samples = samples.shape[0]
        self.dim = samples.shape[1]
        self.points = np.zeros((self.props_per_sample + 1, samples.shape[0], self.dim))
        for i in range(samples.shape[0]):
            self.points[0, i, :] = samples[i, :]
            if i < samples.shape[0] - 1:
                self.points[1:, i, :] = proposals[
                    (i * self.props_per_sample):((i+1) * self.props_per_sample),
                    :
                ]

        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        self.proposal_line = ax.plot([], [])[0]
        self.sample_line = ax.plot((), (), linestyle='', marker='.')[0]

        self.anim = animation.FuncAnimation(
            fig, lambda i: self.animate(i), 
            self.points.shape[0] * self.points.shape[1], interval=20, blit=True, repeat_delay=1000
        )

    def show(self):
        plt.show()

    def animate(self, i):
        sample_i, prop_i, = np.unravel_index(i, (self.n_samples, self.props_per_sample + 1))

        if sample_i < self.n_samples - 1:
            self.proposal_line.set_data(
                self.points[0:prop_i, sample_i, 0], 
                self.points[0:prop_i, sample_i, 1]
            )
        self.sample_line.set_data(
            self.points[0, 0:sample_i + 1, 0], 
            self.points[0, 0:sample_i + 1, 1]
        )
        return [self.proposal_line, self.sample_line]

    def save(self, filename):
        self.anim.save(filename, writer="ffmpeg")
