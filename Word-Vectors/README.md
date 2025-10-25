# Word2vec
Framework for learning word vectors
Every word in a fixed vocabulary is represented by a vector.
Calculate the probability of a word given the context (surrounding words), or vice versa, given a word calculate the probability of the context words.

In the end, it's calculating the probability with certain score: observed - expected

# Optimization: Gradient Descent
We have a cost function, J, that we want to minimize: the probability of words fitting in the context, but since we added a minus, it becomes inverted and we want to minimize instead of maximize.

Gradient Descent is an algorithm to minimize the function. Basically iterates over it. Calculate the current value of function, take small step in direction of negative gradient and repeat.

The problem is that the function is for all windows of the corpus (potentially billions). So calculating the gradient is very expensive to compute. The solution is Stochastic Gradient Descent: repeatedly sample windows, and update after each one.

Mini Batch Gradient Descent

```python
while True:
    window = sample_window(corpus)
    theta_grad = evaluate_gradient(J, window, theta)
    theta = theta - alpha * theta_grad
```
alpha is the learning delta.

# Model variants:
- Skip-grams (SG): predict context ("outside") words (position independent) given a center word
- Continuous Bag of Words (CBOW): predict center word from (bag of) context words.

# Loss functions for training:
- Naive Softmax (simple but expensive loss function, when many output classes)
- More optimized variants like hierarchical softmax.
- Negative sampling

Idea: train binary logistic regressions to differenciate a true pair (center word and a word in its context window) versus several "noise" pairs (the center word paired with a random word) -> This would be the negative sampling!

We take K negative samples (using word probabilities). Maximize probability of real outside word; minimize the probability of random words.
Using logistic/sigmoid function (instead of softmax) can also simply the process: it is used for multi-label and binary classification. Whereas, softmax is used when outputs are mutually exclusive, so the probabilities of the classification sum to 1.

There are few simplication that help to reduce the complexity of the problem: ratios of co-occurence probabilities can enconde meaning components -> the occurrence of certain words within certain corpus can cluster them into semantic axes. So instead of analyzing the co-occurrence of all words, we calculate the co-occurence of words into semantic groups!

GloVe: encoding meaning components in vector differences. We can to capture them as linear meaning components in a word vector space.

# How to evaluate word vectors?
General concept of evaluation, in NLP: intrinsic vs extrinsic.
- Intrinsic:
    - Evaluation on a specific/intermediate subtask
    - Fast to compute
    - Helps to understand that system
    - Not clear if really helpful unless correlation to real task is established
- Extrinsic:
    - Evaluation on a real task
    - Can take a long time to compute accuracy?
    - Unclear if the subsystem is the problem or its interaction or other subsystems
    - If replacing exactly one subsystem with another improves accuracy -> winning!

# Word Senses
Different senses of a word reside in a linear superposition (weighted sum) in standard word embeddings like word2vec.

**SVD (like LSA or COALS)**: represents every word as a dense vector - every latent dimension contribues a little bit to each word's meaning.
    Example: "cat" = 0.3 x animal + 0.2 x pet + 0,1 x mammal + 0.05 x cute + ...

**Sparse coding**: finds a set of semantic building blocks where each word uses only a few of them, and those few explain most of the meaning. Tends to produce clearer, more interpretable features (you can often label each dimension with a meaning like animal, food, vehicle, etc.)
    Example: "cat" = animal + pet (everything else = 0)

Basicalli, with Sparse coding, instead of one fuzzy vector you get a sparse mixture that can shift depending on which sense is used.

Modern contextual models, like BERT and GPT, implicitly do something similar to sparse coding:
- Each layer learns many latent "semantic features".
- Depending on the context, only some of them activate strongly for a given word. A kind of dynamic sparsity.

# Deep Learning classification: Named Entity Recognition (NER)
Cross entropy!

A binary logistic regression unit is a bit similar to a neuron:
f -> nonlinear activation function (e.g. sigmoid), with weights, bias, hidden layer and inputs. Inputs will determine the level of excitation of the neuron to whether get activated or not.

Neural network == running several logistic regressions at the same time. We can feed them into another logistic regression function, giving composed functions. It's the final loss function that will direct what the intermediate hidden variables should be, so as to do a good job at predicting the targets for the next layer, etc.