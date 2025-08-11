import numpy as np


def four_neighborhood(height: int, width: int) -> np.ndarray:
    """
    Construct a 4-neighborhood adjacency for a grid of given dimensions.

    Nodes are indexed row-major from 0 to height*width - 1. Edges are
    bidirectional and returned as a 2 x E array where each column is an
    edge (u, v).

    Args:
        height: Number of rows in the grid.
        width: Number of columns in the grid.

    Returns:
        A NumPy array of shape (2, E) representing the adjacency edges.
    """
    idx = np.arange(height * width).reshape(height, width)
    edges = []
    for i in range(height):
        for j in range(width):
            u = idx[i, j]
            # down
            if i + 1 < height:
                edges.append((u, idx[i + 1, j]))
            # up
            if i - 1 >= 0:
                edges.append((u, idx[i - 1, j]))
            # right
            if j + 1 < width:
                edges.append((u, idx[i, j + 1]))
            # left
            if j - 1 >= 0:
                edges.append((u, idx[i, j - 1]))
    if not edges:
        return np.empty((2, 0), dtype=int)
    # Transpose to shape (2, E)
    return np.array(edges, dtype=int).T
