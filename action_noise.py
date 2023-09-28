import numpy as np

class OUActionNoise():
    # Ornstein-Uhlenbeck noise (Brownian motion)
    def __init__(self, size, sigma=0.2, theta=0.15, dt=1e-2):
        # values for sigma and theta from the original paper
        self.theta = theta
        self.dt = dt
        self.sigma = sigma
        self.size = size
        self.reset()

    def __call__(self):
        # Kloeden, Peter E.; Platen, Eckhard; Schurz, Henri (1994). Numerical solution of SDE through computer experiments.
        x = self.x_prev \
            - self.theta * self.x_prev * self.dt \
            + self.sigma * np.sqrt(2.*self.dt*self.theta) * np.random.normal(size=self.size)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = np.zeros(self.size)