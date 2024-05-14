from sklearn.neighbors import NearestNeighbors
import numpy as np
import time


def get_min_pair(corners):
    """ corners ... Nx2 numpy array of N points
        task: finds closest two points
        returns (distance between the closest two points, index of the first, index of the second)
    """
    min_norm = np.inf
    min_pair = None
    if corners.shape[0] <= 1:
        return (min_norm, None, None)
    for i in range(corners.shape[0]):
        norms = np.linalg.norm(corners - corners[i, :], axis=1)
        norms[i] = np.inf
        mn = np.min(norms)
        if mn < min_norm:
            min_norm = mn
            min_pair = (i, np.argmin(norms))
    return (min_norm, min_pair[0], min_pair[1])


if __name__ == '__main__':
    pts2D = np.random.randn(1000, 2)

    start1 = time.clock()
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(pts2D)
    distances, indices = nbrs.kneighbors(pts2D)
    print(time.clock() - start1)

    start2 = time.clock()
    get_min_pair(pts2D)
    print(time.clock() - start2)
