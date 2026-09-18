
import numpy as np
import math
from scipy.spatial import cKDTree
from numba import njit

# Standalone compiled function
@njit(fastmath=True)
def _euclidean_distance_numba(point1, point2):
    """Numba-compiled core distance calculation"""
    diff = 0.0
    for i in range(point1.shape[0]):
        diff += (point1[i] - point2[i])**2
    return np.sqrt(diff)

class MeshDistanceCalculator:
    def __init__(self):
        pass




    # @njit(fastmath=True)
    # def euclidean_distance_numba(self, point1, point2):
    #     """Ultra-fast version using Numba JIT compilation"""
    #     diff = 0.0
    #     for i in range(len(point1)):
    #         diff += (point1[i] - point2[i])**2
    #     return np.sqrt(diff)

    def euclidean_distance(self, point1, point2):
        """Optimized wrapper for the numba function"""
        # Convert to contiguous numpy arrays (only needed if inputs aren't already arrays)
        # arr1 = np.ascontiguousarray(point1, dtype=np.float64)
        # arr2 = np.ascontiguousarray(point2, dtype=np.float64)
        return _euclidean_distance_numba(np.asarray(point1), np.asarray(point2))
    
    # def euclidean_distance(self, point1, point2):
    #     # Convert points to numpy arrays
    #     point1 = np.array(point1)
    #     point2 = np.array(point2)
    #     # print("point1", point1)
    #     # print("point2", point2)
    #     # Reshape points if necessary
    #     # if point1.ndim == 1:
    #     vertices_num = int(len(point1) / 3)
    #     point1 = point1.reshape((vertices_num, 3))
    #     point2 = point2.reshape((vertices_num, 3))

    #     # Calculate Euclidean distance
    #     distance = np.sqrt(np.sum((point1 - point2)**2))

    #     return distance

    def chamfer_distance(self, point1, point2):
        vertices_num = int(len(point1) / 3)
        point1 = np.reshape(point1, (vertices_num, 3))
        point2 = np.reshape(point2, (vertices_num, 3))

        tree1 = cKDTree(point1)
        tree2 = cKDTree(point2)

        distances1, _ = tree1.query(point2)
        distances2, _ = tree2.query(point1)

        chamfer_dist = np.mean(distances1) + np.mean(distances2)
        return chamfer_dist

    def bulldozer_distance(self, point1, point2):
        vertices_num = int(len(point1) / 3)
        point1 = np.reshape(point1, (vertices_num, 3))
        point2 = np.reshape(point2, (vertices_num, 3))

        tree1 = cKDTree(point1)
        distances, _ = tree1.query(point2)
        bulldozer_dist = np.max(distances)
        return bulldozer_dist

    def hausdorff_distance(self, point1, point2):
        vertices_num = int(len(point1) / 3)
        point1 = np.reshape(point1, (vertices_num, 3))
        point2 = np.reshape(point2, (vertices_num, 3))

        tree1 = cKDTree(point1)
        tree2 = cKDTree(point2)

        distances1, _ = tree1.query(point2)
        distances2, _ = tree2.query(point1)

        hausdorff_dist = max(np.max(distances1), np.max(distances2))
        return hausdorff_dist

    def rms_distance(self, point1, point2):
        vertices_num = int(len(point1) / 3)
        point1 = np.reshape(point1, (vertices_num, 3))
        point2 = np.reshape(point2, (vertices_num, 3))

        squared_diff = np.sum((point1 - point2) ** 2, axis=1)
        rms_dist = np.sqrt(np.mean(squared_diff))
        return rms_dist

    def closest_point_distance(self, point1, point2):
        vertices_num = int(len(point1) / 3)
        point1 = np.reshape(point1, (vertices_num, 3))
        point2 = np.reshape(point2, (vertices_num, 3))

        tree1 = cKDTree(point1)
        tree2 = cKDTree(point2)

        distances1, _ = tree1.query(point2)
        distances2, _ = tree2.query(point1)

        closest_dist1 = np.mean(distances1)
        closest_dist2 = np.mean(distances2)
        return closest_dist1, closest_dist2

    def calculate_distance(self, method, point1, point2):
        if method == 'euclidean':
            return self.euclidean_distance(point1, point2)
        elif method == 'chamfer':
            return self.chamfer_distance(point1, point2)
        elif method == 'bulldozer':
            return self.bulldozer_distance(point1, point2)
        elif method == 'hausdorff':
            return self.hausdorff_distance(point1, point2)
        elif method == 'rms':
            return self.rms_distance(point1, point2)
        elif method == 'closest':
            return self.closest_point_distance(point1, point2)
        else:
            raise ValueError("Unknown distance method: {}".format(method))

