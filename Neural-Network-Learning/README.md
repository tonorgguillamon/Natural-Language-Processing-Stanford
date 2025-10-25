# Neural Network Learning
Non-linearities:
- logistic ("sigmoid")
- tanh
- hard tanh (simplified version of tanh)
- ReLU (Rectified Linear Unit)
- Leaky ReLU/Parametric ReLU
- GELU and Swish -> often used with Transformers (BERT, RoBERTa, etc.)

Neural Networks do function approxiamtion (regression or classification).
Without non-linearities, deep neural networks can't do anything more than a linear transform. Extra layers could just be compiled down into single linear transfor. But, with more layers that include non-linearities, they can approximate any complex function!

## Gradients
Matrix calculus: fully vectorized gradients.

## Backpropagation
It's taking derivatives and using chain rule. We re-use derivatives computed for higher layers in computing derivatives for lower layers to minimize computation.

Forward propagation:
```
x ----> * ----> + ---> f ----> * ---->
        ^       ^              ^
        |       |              |
        W       b              u
```

Backpropagation is going backwards, in previous function, and calculate the derivatives. We pass along gradients.

Node: receives an "upstream gradient". Goal is to pass on the correct "downstream gradient". So each node has a local gradient: gradient of its output with respect to its input.
For multiple inputs -> multiple local gradients.

downstream gradient = upstream gradient x local gradient

Upwards, from s (output) to b, it's always the same computation. Hence, reuse those gradients, and only reevaluate W.

## Dependency parsing
Sources of information:
- Bilexical affinities: the dependency [discussion -> issues] is plausible
- Dependency distance: most (but not all) dependencies are between nearby words
- Intervening material: dependencies rarely span intervening verbs or punctuatio 
- Valency of heads: how many dependens on which side are usual for a head?

### MaltParser
Tool used to find the grammatical structure (syntax) of sentences. It's a dependency parser, meaning it shows which words depend on which others in a sentence.

1. Input: a sentency that's already split into words (tokenized) and sometimes tagged with parts of speech (like noun, verb, etc)
2. Parsing: MaltParser uses a machine learning model (trained from example sentences) to decide which words depend on which others
3. Output: a tree structure showing dependencies (who depends on whom)

It helps understand sentence meaning better; chatbots, translation systems and information extraction tools; and analyze linguistic structure for research

```python
from nltk.parse import malt
mp = malt.MaltParser('path/to/maltparser-1.9.2.jar', 'path/to/engmalt.linear-1.7.mco')
tree = mp.parse_one(['The', 'cat', 'sat', 'on', 'the', 'mat'])
print(tree)
```

Downside: sparse, incomplete and expensive to compute.
More than 95% of parsing time is consumed by feature computation.

So...NEURAL APPROACH: learn a dense and compact feature representation!
DL classifiers are non-linear: they can learn much more complex nonlinear decision boundaries.

# Key points:
U: [embed size x num tokens]
V: [embed size x num tokens]

gradient dJ/dvc -> [embed size x 1]

gradien dJ/dU -> [embed size x num tokens]

In naive softmax, the gradient is dependent on summation across all classes. In the case of word2vec,
this means summing across all the words in the vocabulary (as can be seen in the above equation). This
tends to slow down the training process.

In negative sampling, computation is cheaper. Instead of summing across all words (represented by uw
in the above equation) we only sum across a few context/negative words(termed as negative samples).