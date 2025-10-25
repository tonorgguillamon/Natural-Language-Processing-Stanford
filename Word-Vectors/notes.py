"""
word2vec and co-occurrence rely on distributional hypothesis. Words that occur in similar contexts tend to have similar meanings.
But they differ in how they learn those relationships.

*co-occurrence*
Pros:
- simple, interpretable
- can be computed directly from stadistics
- captures global structue of corpus (all co-occurrences at once)
Cons:
- matrix can be huge
- doesn't generalize easily to new words or contexts
- slow for very large corpora

*word2vec*
Learns embeddings through prediction, NOT counting.
Instead of storing co-occurrence counts, it trains a neural network to: predict context words given a target (skip-gram) or predict a target given a context words (cbow)
Pros:
- learns, dense, compact embeddings
- efficient on larga corpora
- captures nonlinear relationships and generalizes well
- empirically strong (basis for GloVe, FastText)
Cons:
- less interpretable mathematically
- requires training (hypeparameters, stochasticity)
- harder to connect to exact co-ocurrence counts

Even though they look very different, word2vec and co-occurrence methods are mathematically related.
Word2vec implicitly factorizes a co-occurrence matrix weighted by log probabilities (PMI).
So word2vec ≈ a smarter, weighted SVD on a co-occurrence matrix, learned indirectly through prediction.
"""