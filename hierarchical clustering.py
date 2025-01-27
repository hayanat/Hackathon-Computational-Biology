import numpy as np
import math
from collections import defaultdict
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import csv

class ClusterManager:
    _registry = defaultdict()

    def __init__(self, key=None):
        self.cluster_id = len(ClusterManager._registry)
        ClusterManager._registry[self.cluster_id] = self
        self.members = []
        if key is not None:
            self.members.append(key)

    def add_members(self, other):
        self.members.extend(other.members)

    def member_count(self):
        return len(self.members)

    def destroy(self):
        if self.cluster_id in ClusterManager._registry:
            del ClusterManager._registry[self.cluster_id]

    @staticmethod
    def current_clusters():
        return [c for c in ClusterManager._registry.values()]

    @staticmethod
    def lookup(cid):
        return ClusterManager._registry.get(cid, None)


class HierarchicalClustering:
    def __init__(self, data_list, alignment_matrix_path, linkage_heuristic="Centroid"):
        self.data = data_list
        self.alignment_matrix_path = alignment_matrix_path
        self.linkage_heuristic = linkage_heuristic

        self._create_initial_clusters()
        self.num_initial = len(ClusterManager.current_clusters())

        self.dist_matrix = np.loadtxt(self.alignment_matrix_path, delimiter=",", skiprows=1, usecols=range(1, len(data_list)+1))

        self.linkage_records = np.zeros((self.num_initial - 1, 4))
        self.cluster2scipy = {}
        for i, cluster in enumerate(ClusterManager.current_clusters()):
            self.cluster2scipy[cluster.cluster_id] = i

        self.next_scipy_id = self.num_initial

    def _create_initial_clusters(self):
        ClusterManager._registry.clear()
        for point in self.data:
            ClusterManager(point)

    def run_clustering(self):
        iteration = 0
        while len(ClusterManager.current_clusters()) > 1:
            cA, cB, min_dist = self._get_min_dist_pair()
            self._merge_clusters(keep_id=min(cA, cB), remove_id=max(cA, cB), iteration_idx=iteration, dist=min_dist)
            iteration += 1

    def _get_min_dist_pair(self):
        current = ClusterManager.current_clusters()
        if len(current) < 2:
            return None, None, 0.0

        active_ids = [c.cluster_id for c in current]
        best_val = math.inf
        best_pair = (None, None)

        for i in range(len(active_ids)):
            for j in range(i + 1, len(active_ids)):
                idA = active_ids[i]
                idB = active_ids[j]
                dist_val = self.dist_matrix[idA, idB]
                if dist_val < best_val:
                    best_val = dist_val
                    best_pair = (idA, idB)
        return best_pair[0], best_pair[1], best_val

    def _merge_clusters(self, keep_id, remove_id, iteration_idx, dist):
        cluster_keep = ClusterManager.lookup(keep_id)
        cluster_remove = ClusterManager.lookup(remove_id)
        size_k = cluster_keep.member_count()
        size_r = cluster_remove.member_count()

        if self.linkage_heuristic == "Centroid":
            weighted_avg = (
                self.dist_matrix[keep_id, :] * size_k + self.dist_matrix[remove_id, :] * size_r
            ) / (size_k + size_r)
        elif self.linkage_heuristic == "Max":
            row_stack = np.vstack(
                (self.dist_matrix[keep_id, :], self.dist_matrix[remove_id, :]))
            weighted_avg = np.amax(row_stack, axis=0)
        elif self.linkage_heuristic == "Min":
            row_stack = np.vstack(
                (self.dist_matrix[keep_id, :], self.dist_matrix[remove_id, :]))
            weighted_avg = np.amin(row_stack, axis=0)

        self.dist_matrix[keep_id, :] = weighted_avg
        self.dist_matrix[:, keep_id] = weighted_avg

        self.dist_matrix[remove_id, :] = math.inf
        self.dist_matrix[:, remove_id] = math.inf

        scipyA = self.cluster2scipy[keep_id]
        scipyB = self.cluster2scipy[remove_id]

        self.linkage_records[iteration_idx, 0] = scipyA
        self.linkage_records[iteration_idx, 1] = scipyB
        self.linkage_records[iteration_idx, 2] = dist
        self.linkage_records[iteration_idx, 3] = size_k + size_r

        new_scipy_id = self.next_scipy_id
        self.next_scipy_id += 1

        self.cluster2scipy[keep_id] = new_scipy_id
        del self.cluster2scipy[remove_id]

        cluster_keep.add_members(cluster_remove)
        cluster_remove.destroy()

        return cluster_keep.member_count()

    def plot_dendrogram(self, title):
        plt.title(title)
        dendrogram(
            self.linkage_records,
            labels=np.array(self.data),
            show_leaf_counts=True,
            show_contracted=True
        )
        plt.show()


def build_tree(alignment_data_path, data_list, matrix_type, linkage_heuristic="Centroid"):
    model = HierarchicalClustering(data_list, alignment_data_path, linkage_heuristic)
    model.run_clustering()
    title = f"Hierarchical Clustering ({matrix_type})"
    model.plot_dendrogram(title)


if __name__ == "__main__":
    matrices = [
        ("dna_time_matrix.csv", "DNA Time Matrix"),
        ("codon_time_matrix.csv", "Codon Time Matrix"),
        ("protein_time_matrix.csv", "Protein Time Matrix")
    ]

    species = ["Human", "cat", "pigeon", "macaques", "Mouse", "Orangutan", "Rat"]

    for matrix_file, matrix_type in matrices:
        for linkage in ["Min"]:
            build_tree(matrix_file, species, matrix_type, linkage_heuristic=linkage)
