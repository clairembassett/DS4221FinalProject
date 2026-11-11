"""microvector. REFERENCE IMPLEMENTATION from Lab 9 - do not edit.

Shipped complete so Lab 10 can build on it: micro_rag.py uses
BruteForceIndex for retrieval (24 docs need no IVF; swap one line if your
project corpus grows).

Lab 9 of Advanced Databases for Data Science (DS 6XXX, Fall 2026).
Pure standard library; vectors are plain Python lists.

THE PROBLEM: given a query vector, find the k most similar stored vectors.
Exact search compares against ALL of them - O(n) dot products per query.
The fix is the same shape as week 6's: a structure that lets you skip most
of the data. Here it's IVF (inverted file) - cluster the vectors, search
only the clusters nearest the query:

    build:  k-means the vectors into C cluster lists (coarse "buckets")
    search: find the P nearest cluster centroids ("probe" P lists),
            then brute-force only inside those lists

    exact recall costs n comparisons; IVF costs ~ C + P*(n/C)
    - and pays for it in RECALL: the true neighbor might live in a
    list you didn't probe. recall@k measures how often it doesn't.

Everything here scales to the real thing: FAISS's IndexIVFFlat and
pgvector's ivfflat are this file with SIMD and better k-means.

Run the tests any time:   python3 test_vector.py
Run the measurement:      python3 measure_recall.py    (after tests pass)
"""

from __future__ import annotations

import heapq
import math
import random


# ---------------------------------------------------------------- basics

def dot(a: list[float], b: list[float]) -> float:
    """Similarity. Our vectors are unit-normalized, so the dot product IS
    cosine similarity: 1.0 = identical direction, 0 = unrelated. Provided."""
    return sum(x * y for x, y in zip(a, b))


def normalize(v: list[float]) -> list[float]:
    """Scale to unit length (so dot = cosine). Provided."""
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


# ---------------------------------------------------------------- exact

class BruteForceIndex:
    """The exact baseline: compare the query with everything.
    Correct by construction, O(n) per query - the thing to beat."""

    def __init__(self, vectors: list[list[float]]):
        self.vectors = vectors
        self.comparisons = 0            # counted for the measurement


    def search(self, query: list[float], k: int) -> list[tuple[float, int]]:
        """The k most similar vectors, best first, ties by id."""
        scored = []
        for i, v in enumerate(self.vectors):
            self.comparisons += 1
            scored.append((dot(query, v), i))
        return heapq.nsmallest(k, scored, key=lambda t: (-t[0], t[1]))



# ---------------------------------------------------------------- IVF

def kmeans(vectors: list[list[float]], n_clusters: int,
           iters: int = 5, seed: int = 6042) -> list[list[float]]:
    """A deliberately small k-means: seeded start, few iterations - good
    enough centroids for bucketing. Provided complete (read it: it's the
    same assign/average loop scikit-learn polishes)."""
    rng = random.Random(seed)
    centroids = [list(v) for v in rng.sample(vectors, n_clusters)]
    for _ in range(iters):
        buckets: list[list[list[float]]] = [[] for _ in range(n_clusters)]
        for v in vectors:
            best = max(range(n_clusters), key=lambda c: dot(v, centroids[c]))
            buckets[best].append(v)
        for c, bucket in enumerate(buckets):
            if bucket:
                dim = len(bucket[0])
                mean = [sum(v[d] for v in bucket) / len(bucket) for d in range(dim)]
                centroids[c] = normalize(mean)
    return centroids


class IVFIndex:
    """Inverted-file index: vectors bucketed by nearest centroid;
    queries probe only the nearest buckets."""

    def __init__(self, vectors: list[list[float]], n_clusters: int = 20):
        self.vectors = vectors
        self.centroids = kmeans(vectors, n_clusters)
        self.lists: list[list[int]] = [[] for _ in range(n_clusters)]
        self.comparisons = 0
        self._build()


    def _build(self) -> None:
        """Assign every vector id to the list of its nearest centroid."""
        for i, v in enumerate(self.vectors):
            best = max(range(len(self.centroids)),
                       key=lambda c: dot(v, self.centroids[c]))
            self.lists[best].append(i)

    def search(self, query: list[float], k: int,
               probe: int = 1) -> list[tuple[float, int]]:
        """Approximate top-k: score the centroids, take the `probe` best
        lists, brute-force only the ids inside them.

        Counts centroid dots and candidate dots alike."""
        centroid_scores = []
        for c, centroid in enumerate(self.centroids):
            self.comparisons += 1
            centroid_scores.append((dot(query, centroid), c))
        nearest_lists = [c for _, c in
                         heapq.nsmallest(probe, centroid_scores,
                                         key=lambda t: (-t[0], t[1]))]
        scored = []
        for c in nearest_lists:
            for i in self.lists[c]:
                self.comparisons += 1
                scored.append((dot(query, self.vectors[i]), i))
        return heapq.nsmallest(k, scored, key=lambda t: (-t[0], t[1]))



def recall_at_k(approx: list[tuple[float, int]],
                exact: list[tuple[float, int]], k: int) -> float:
    """What fraction of the true top-k did the approximate search find?
    Provided - the honesty metric every ANN system reports."""
    truth = {i for _, i in exact[:k]}
    found = {i for _, i in approx[:k]}
    return len(truth & found) / k
