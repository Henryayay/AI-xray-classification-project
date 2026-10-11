"""In-context exemplar selection strategies.

* ``random``: class-balanced random exemplars, fixed for all queries within a
  draw (as in the original notebooks), re-drawn independently per draw.
* ``knn``: retrieval-augmented ICL. For each query, the ``k/2`` training images
  of each class that are most similar (cosine, frozen encoder embeddings) are
  used. The draw seed controls only exemplar order, so draw-to-draw variance
  isolates order sensitivity.

``max_similarity`` excludes candidates whose cosine similarity to the query
exceeds a threshold, so near-duplicate copies of the query (see the leakage
audit) cannot be retrieved as exemplars.

In both cases exemplars are shuffled so the label sequence carries no
positional signal, and the training split is the only exemplar source.
"""

import numpy as np


class ExemplarSelector:
    def __init__(self, strategy, train_labels, shots, seed, train_emb=None, max_similarity=None):
        if shots % 2:
            raise ValueError("shots must be even for class balance")
        self.strategy = strategy
        self.labels = np.asarray(train_labels)
        self.k = shots // 2
        self.seed = seed
        self.by_class = {c: np.flatnonzero(self.labels == c) for c in (0, 1)}
        self.max_similarity = max_similarity
        if strategy == "random":
            rng = np.random.default_rng(seed)
            picks = np.concatenate([rng.choice(self.by_class[c], self.k, replace=False)
                                    for c in (0, 1)])
            rng.shuffle(picks)
            self.fixed = picks
        elif strategy == "knn":
            if train_emb is None:
                raise ValueError("knn strategy requires train embeddings")
            from .embed import l2_normalise

            self.train_emb = l2_normalise(np.asarray(train_emb, dtype=np.float32))
        else:
            raise ValueError(f"unknown strategy {strategy}")

    def select(self, query_emb=None, query_key=0):
        """Training indices of exemplars for one query (deterministic given seed and key)."""
        if self.k == 0:
            return np.array([], dtype=int)
        if self.strategy == "random":
            return self.fixed
        q = np.asarray(query_emb, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        sims = self.train_emb @ q
        picks = []
        for c in (0, 1):
            idx = self.by_class[c]
            if self.max_similarity is not None:
                idx = idx[sims[idx] < self.max_similarity]
            picks.append(idx[np.argsort(-sims[idx], kind="stable")[: self.k]])
        picks = np.concatenate(picks)
        rng = np.random.default_rng([self.seed, int(query_key)])
        rng.shuffle(picks)
        return picks
