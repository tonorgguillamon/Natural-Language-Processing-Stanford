# LLMs/RNNs/LSTMs
## Introduction
- Regularization
- Dropout: drop inputs in hidden layers of the neural network (usually by adding a mask-layer). The model needs to learn knowing that sometimes some parameters/attributes might just not be present. Thus, it avoids feature co-adaptation, resulting in a good regularization.
- Vectorization: avoid using for loops. Using matrices is faster even running only on the CPU.
- Parameter initialization: you normally must initialize weights to small random values (i.e., NOT zero matrices!). To avoid symmetries that prevent learning/specialization.
- Optimizers: usually, plain SGD will work just fine. For more complex nets you can try more sophisticated "adaptative" optimizers that scale the adjustment to invididual parameters by an accumulated gradient (e.g. Adagrad, Adam, NAdamW)

## LLM
Language Modeling is the task of predicting what word comes next. In other words: a system that assigns probability to a piece of text.
Sparsity Problems:
- when a really specific combination of words never appeared in the corpus, then that probability is going to be 0 even if it could be realistic. Solution:
* add an small alpha to the count for every word in the vocabulary -> smoothing
* take less word from the condition to guess next words   -> backoff (maybe: "I took a bus during" was never on the data, but "I took a bus" could have been)

## RNN-LM
Basically apply the same weights repeatedly, for every word.
Advantages:
- can process any length input
- computation for step t can use information from many steps back
- model size doesn't increase for longer input context (more computation tho)
- same weights applied on every timestep, so there is symmetry in how inputs are processed
Disadvantage:
- computation is slow! it's sequential, one vector at a time
- in practise, difficult to access information from many steps back
- forgets information back-in-time

![training-rnn](resources/training-rnn.png)

### Metrics
* Entropy measures how uncertain the model’s predictions are
- Low entropy → the model is confident (probabilities are concentrated on one token).
- High entropy → the model is uncertain (probabilities are spread out).
* Perplexity is simply the exponentiation of entropy. The average number of choices the model is “confused” between
- If perplexity = 1 → perfectly certain (always correct).
- If perplexity = 10 → the model is, on average, as uncertain as choosing between 10 equally likely words.
* Gradient is used to reduce entropy and perplexity by updating parameters. It doesn’t measure uncertainty — it reacts to it and changes the weights to make the model less uncertain (i.e., reduce entropy → reduce perplexity → lower loss).

## LSTM
LSTM architecture makes it much easier for an RNN to preserve information over many timesteps.
e.g.: if the forget gate is set to 1 for a cell dimension and the input gate set to 0, then the information of that cell is preserved indefinitely.
Usually, a LSTM remember the last 7 words.

An LSTM carries out:
- Forget some cell content
- Write some new call content, coming from previous computations.
- Outputs some cell content to the hidden state.

Main issues of LSTM: exploting and vanishing.

Possible solutions:
* Add more direct connections, thus allowing the gradient to flow -> residual connections (ResNet): preserves information by default, and makes deep networks much easier to train.
* Dense connections (DenseNet): directly connect each layer to all future layers.
* Highway connections (HighwayNet): similar to residual connections, but the identity connection vs the transformation layer is controller by a dynamic gate.

## Multi-layer RNNs
Allow a network to compute more complex representations. They work better than just have one layer of high-dimensional encodings. The lower RNNs should compute lower-level features and the higher RNNs should compute higher-level features.
High-performing RNNs are usually multi-layers, but aren't as deep as convolutional or feed-forward networks.

## Neural Machine Translation
It's a way to do Machine Translation with a single end-to-end neural network. The neural network architecture is called a sequence-to-sequence model (aka seq2seq) and it involves two RNNs.

```
Encoder RNN   --------------------------------> <START> + ... + <END> -------------------> Decoder RNN
(encoding of the source sentence                                           (generates target sentence, conditioned on encoding)
provides initial hidden state for decoder RNN)
```

Basically, one neural network takes input and produces a neural representation. Another network produces output based on that neural representation.
seq2seq model is an example of a Conditional Language Model.

# Attention
Seq2seq has a bottleneck problem: the encoding of the source sentence has to capture all information about the source sentence. This is an information bottleneck!

Attention is the solution: on each step of the decoder, use direct connection to the encoder to focus on a particular part of the source sequence.

In each word, calculate the attention scores, then obtain the attention distributions (take softmax to turn the scores into a probability distribution) and finally generate attention output. This returns the next word. With next word predicted, calculate again the attention scores with the source sentence + new word. Then again the attention distribution and the attention output to predict the second word.

## How to compute attention scores
1. Basic dot-product attention. This assumes d1 = d2 (input = output)
2. Multiplicative attention. Adding a weight matrix in between. "Bilinear attention". Problem is that the matrix might get really large, since d1xd2.
3. Reduced-rank multiplicative attention. Takes two skinny matrixes and multiple them together. This way the weigths to learn are smaller. Playing a bit witht the matrix, that architecture would be the same as [skinny1 x d1]T x [skinny2 x d2]
4. Additive attention. Uses a feed-forward neural net layer.