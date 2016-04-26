"""
Benchmark a game of life implementation.
"""

import numpy as np

from numba import jit


@jit(nopython=True)
def wrap(k, max_k):
    if k == -1:
        return max_k - 1
    elif k == max_k:
        return 0
    else:
        return k

@jit(nopython=True)
def increment_neighbors(i, j, neighbors):
    ni, nj = neighbors.shape
    for delta_i in (-1, 0, 1):
        neighbor_i = wrap(i + delta_i, ni)
        for delta_j in (-1, 0, 1):
            if delta_i != 0 or delta_j != 0:
                neighbor_j = wrap(j + delta_j, nj)
                neighbors[neighbor_i, neighbor_j] += 1

@jit
def numba_life_step(X):
    # Compute # of live neighbours per cell
    neighbors = np.zeros_like(X, dtype=np.int8)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i,j]:
                increment_neighbors(i, j, neighbors)
    # Return next iteration of the game state
    return (neighbors == 3) | (X & (neighbors == 2))


start_state = np.random.RandomState(0).random_sample((300, 200)) > 0.5

def run_game(nb_iters):
    state = start_state
    for i in range(nb_iters):
        state = numba_life_step(state)
    return state


def setup():
    """
    Precompile jitted functions.
    """
    run_game(10)


class GameOfLife:

    def time_gameoflife(self):
        run_game(10)

