import numpy as np


class PageRank:
    def __init__(self, random_walk, teleportation):
        self.random_walk = random_walk
        self.teleportation = teleportation
        self.identity_matrix = np.identity(len(random_walk), dtype=float)
        self.ppr = None
        self.final_steady_state = None
        self.page_rank()

    def page_rank(self):
        self.ppr = self.identity_matrix - self.random_walk
        steady_state = np.linalg.inv(self.ppr)
        self.final_steady_state = steady_state.dot(self.teleportation)

    def get_final_steady_state(self):
        return self.final_steady_state





