# pulse_db_clustering.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# ------------------------------
# 1. Data Loader
# ------------------------------
class TimeSeriesLoader:
    def __init__(self, file_path=None):
        self.data = None
        if file_path:
            self.data = self.load_csv(file_path)

    def load_csv(self, path):
        """
        Load time-series data from a CSV file.
        Each row corresponds to one 10-second segment.
        """
        return np.loadtxt(path, delimiter=',')


# ------------------------------
# 2. Divide-and-Conquer Clustering
# ------------------------------
class DivideConquerClustering:
    def __init__(self, similarity_threshold=0.9, min_cluster_size=2):
        self.threshold = similarity_threshold
        self.min_size = min_cluster_size

    def cluster(self, segments):
        """
        Perform recursive divide-and-conquer clustering.
        segments: list or array of time series
        """
        if len(segments) <= self.min_size:
            return [segments]

        # Compute pairwise similarity matrix (correlation)
        n = len(segments)
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                corr = np.corrcoef(segments[i], segments[j])[0, 1]
                sim_matrix[i, j] = corr
                sim_matrix[j, i] = corr

        # Simple split: divide into two clusters by median similarity to first segment
        first_segment = segments[0]
        sims = [np.corrcoef(first_segment, s)[0,1] for s in segments]
        cluster1 = [segments[i] for i in range(n) if sims[i] >= self.threshold]
        cluster2 = [segments[i] for i in range(n) if sims[i] < self.threshold]

        clusters = []
        if cluster1:
            clusters += self.cluster(cluster1)
        if cluster2:
            clusters += self.cluster(cluster2)
        return clusters


# ------------------------------
# 3. Closest Pair Finder
# ------------------------------
class ClosestPair:
    def __init__(self, metric='DTW'):
        self.metric = metric

    def find_closest(self, cluster):
        """
        Find closest pair of time series in cluster using DTW
        """
        min_dist = float('inf')
        pair = (None, None)
        n = len(cluster)
        for i in range(n):
            for j in range(i + 1, n):
                if self.metric == 'DTW':
                    dist, _ = fastdtw(cluster[i], cluster[j], dist=euclidean)
                else:
                    dist = np.linalg.norm(cluster[i]-cluster[j])
                if dist < min_dist:
                    min_dist = dist
                    pair = (cluster[i], cluster[j])
        return pair, min_dist


# ------------------------------
# 4. Maximum Subarray Analyzer
# ------------------------------
class MaxSubarrayAnalyzer:
    def kadane(self, series):
        """
        Kadane's algorithm: returns max sum and interval (start, end)
        """
        max_sum = -float('inf')
        current_sum = 0
        start = end = s = 0
        for i, val in enumerate(series):
            current_sum += val
            if current_sum > max_sum:
                max_sum = current_sum
                start = s
                end = i
            if current_sum < 0:
                current_sum = 0
                s = i + 1
        return max_sum, (start, end)


# ------------------------------
# 5. Report Generator
# ------------------------------
class ReportGenerator:
    def generate_report(self, clusters, closest_pairs, max_intervals):
        print("Number of clusters:", len(clusters))
        for idx, cluster in enumerate(clusters):
            print(f"\nCluster {idx+1}: Size = {len(cluster)}")
            print("Closest pair distance:", closest_pairs[idx][1])
            print("Max subarray intervals (first segment):", max_intervals[idx][0])

            # Plot first two segments of the cluster
            plt.figure(figsize=(8,4))
            plt.plot(cluster[0], label='Segment 1')
            if len(cluster) > 1:
                plt.plot(cluster[1], label='Segment 2')
            plt.title(f'Cluster {idx+1} Example Segments')
            plt.legend()
            plt.show()


# ------------------------------
# 6. Toy Example Verification
# ------------------------------
def toy_example():
    # Toy time-series
    ts1 = np.array([1, 2, 3, 1, 0])
    ts2 = np.array([2, 3, 4, 2, 1])
    ts3 = np.array([10, 11, 10, 9, 8])
    segments = [ts1, ts2, ts3]

    print("Running toy example...")

    # Clustering
    clustering = DivideConquerClustering(similarity_threshold=0.9)
    clusters = clustering.cluster(segments)
    print(f"Clusters formed: {len(clusters)}")

    # Closest pair
    closest_pair_finder = ClosestPair()
    closest_pairs = []
    for cluster in clusters:
        pair, dist = closest_pair_finder.find_closest(cluster)
        closest_pairs.append((pair, dist))

    # Max subarray
    analyzer = MaxSubarrayAnalyzer()
    max_intervals = []
    for cluster in clusters:
        cluster_intervals = [analyzer.kadane(s) for s in cluster]
        max_intervals.append(cluster_intervals)

    # Report
    reporter = ReportGenerator()
    reporter.generate_report(clusters, closest_pairs, max_intervals)

# ------------------------------
# 7. Main Execution
# ------------------------------
if __name__ == "__main__":
    toy_example()

    # For real PulseDB dataset:
    # loader = TimeSeriesLoader("pulse_db_segments.csv")
    # segments = loader.data
    # clustering = DivideConquerClustering(similarity_threshold=0.9)
    # clusters = clustering.cluster(segments)
    # ...
