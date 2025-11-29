# Pretraining
### Byte-pair encoding
If a word is very common it ends up in the vocabulary. Words that don't seem to be "real" words are usually split by patterns from real words.
Example:
- taaaaasty -> vocab mapping: taa## aaa## sty -> 3 embeddings
- Transformerify -> vocab mapping: Transformer## ify -> 2 embeddings

### Introduction
In model NLP all parameters networks are initialized via pretraining. Pretraining methods hide parts of the inputs from the model and train the model to reconstruct those parts.

1. Pretrain (on language modeling): lots of text, learn general things.
2. Finetune (on your task): not many labels, adapt to the task.

### Types of architectures:
- Encoders: gets bidirectional context, can condition on future. How do we train them to build strong representations?
- Encoders-Decorders:  good parts of decoders and encoders? what's the best way to pretrain them?
- Decoders: language models. Noce to generatre from, can't condition on future words.

## Pretraining encoders
We can't really do language modeling, since they get bidirectional context.
Idea: replace some fraction of words in the input with a special [MASK] token; predict these words. Only add loss terms from words that are "masked out".

Examples: BERT, RoBERTa. Good for classification, sentiment, filling up gaps. Bad for text generation or sumarization.

#### Full Finetuning vs Parameter-Efficient Finetuning
Full Finetuning is memory-intensive. Lightweight finetuning trains a few existing or new parameters. More efficient and good enough.

- Prefix-tuning: adds a prefix of parameters and freezes all pretrained parameters. The prefix is processed by the model just like real words would be. Each element of a batch at inference could run a different tuned model.
- Low-Rank Adaptation: learns a low-rank "diff" between the pretrained and finetuned weight matrices. It's easier to learn than prefix-tuning. W + AB -> weights are fixed, A and B are trained.

## Pretraining encoder-decoders
Best way is by Span Corruption: replace different-length spans from the input with unique placeholders; decode out the spans that were removed
Inputs: "Thank you <X> me to your party <Y> week"
Targets: "<X> for inviting <Y> last <Z>"

The decoder learns to fill in the blanks.

Example: T5

## Pretraining Decoders
When using language model pretrained decoders, we can ignore that they were trained to model p(w_t|w_1:t-1). We can finetune them by training a classifier on the last word's hidden state.
```
h1, ..., h_T = Decoder(w1, ..., w_T)
y ~ A h_t + b

Where A and b are randomly initialized and specified by the downstream task.
```

Gradients backpropagate through the whole network.

It's natural to pretrain decoders as language models and then use them as generators, finetuning their p(w_t|w_1:t-1). This is helpful in tasks where the output is a sequence with a vocabulary like that at pretraining time:
- Dialogue (context = dialogue history)
- Summarization (context = document)

In this case:
```
h1, ..., h_T = Decoder(w1, ..., w_T)
w_t ~ A h_t-1 + b

where A and b were pretrained in the language model.
```

## What does pretraining teach?

# Natural Language Generation
For non-open-ended tasks we usually use a enconder-decoder system, where this autoregressive model serves as the decoder, and we'd have another bidirectional encoder for encoding the inputs.
For open-ended tasks (e.g. story generation), this autoregressive generation model is often the only component. Autoregressive = chain-rule -> you generate token_t with all previous tokens. Then token_t+1 with the previous tokens. So it's a chain of predictions.

## Decoding from NLG models
1. At each time step t, out model computes a vector of scores for each token in our vocabulary.
2. Then, we compute a probability distribution over these scores with a softmax function.
3. Our decoding algorithm defines a function to select a token from this distribution.

BUT: this logic has issues with word repeatition. If there is a word that appears a lot in the training set, it just increases the probability of the model choosing that word. The model might enter into a loop of repeating over and over the same words.
Thus, finding the most likely string does't seem the most reasonable approach for open-ended generation.

To solve this:
- Add randomness. Still, this could be problematic because it makes every token in the vocabulary an option. Many tokens are probably really wrong in the current context.
- Choose top-k sampling. Only sample from the top k tokens in the probability distribution. However, top-k sampling can cut off too quickly or too slowly!
- Top-p sampling. Sample from all tokens in the top p cumulative probability mass (i.e. where mass if concentrated). The amount of "k" sampling varies depending on the uniformity of the Probability of the following token.
- Additionally:
	* typical sampling: reweights the score based on the entropy of the distribution
	* epsilon sampling: set a threshold for lower bounding valid probabilities.
- Scaling randomness: temperature. Apply a temperature hyperparameter to the softmax to rebalance P_t. The factor divides the probability of the word, hence the bigger, the less probability.
	* If we increase the temperature > 1, P_t becomes more uniform. More diverse output, probability is spread around vocab.
	* If we decrease the temperature < 1, P_t becomes more spiky. Less diverse output, probability is concentrated on top words.
	
Another issue of decoding: what if I decode a bad sequence from my model? -> Re-rank!
- Decode a bunch of sequences (~10 is a common number)
- Define a score to approximate quality of sequences and re-rank by this score. Simplest is to use (low) perplexity. Re-ranking can score a variety of properties. Eventually we can compose multiple re-rankers together.

## Training NLG models
### Reward Estimation
What behaviours can we tie to rewards?
- Cross-modality consistency in image captioning
- Sentence simplicity
- Temporal Consistency
- Utterance Politeness
- Formality
- Human Preference (RLHF) -> this is the technique behind ChatGPT.

## Evaluation NLG Systems
### N-gram overlap metrics
Word overlap-based metrics such as BLEU, ROUGE, METEOR, CIDEr
They are not ideal for machine translation. They have false positive and false negatives.
They get progressively much worse for tasks that are more open-ended than machine translation.

### Model-based metrics to capture more semantics
Use learned representations of words and sentences to compute semantic similarity between generated and reference texts. No more n-gram bottleneck because text units are represented as embeddings.
The embeddings are pretrained, distance metrics used to measure the similarity can be fixed.
- Vector Similarity: embedding based similarity for semantic distance between text -> embedding average, vector extrema, MEANT, YISI.
- Word Mover's Distance: measures the distance between two sentences (e.g. sentences, paragraphs), using word embedding similarity matching.
- BERTSCORE: uses pre-trained contextual embeddings from BERT and matches words in candidate and reference sentences by cosine similarity.
- Sentence Movers Similarity: evaluate text in continuous space using sentence embeddings from recurrent neural network representations.
- BLEURT: a regression model based on BERT that returns a score that indicates to what extent the candidate text is grammatical and conveys in the meaning of the reference text.

What about Open-ended text generation?
MAUVE: computes information divergence in a quantized embedding space, between the generated text and the gold reference text. It captures distributional similarity, not token overlap.
It's based on computing the divergence curve between two probability distributions using information theory.
Steps:
1. Embed all human sentences and model-generated sentences using a large language model.
2. Cluster these embeddings into K clusters to approximate the underlying probability distributions.
3. Build two discrete distributions over these clusters:
	- P: distribution of human texts
	- Q: distribution of model-generated texts
4. Compute a tradeoff curve of:
	- How much P diverges from Q
	- Versus how much Q diverges from P
5. Compute the area under this divergence curve.
This area is the MAUVE score!
High MAUVE (close to 1) -> generated text distribution very similar to human text
Low MAUVE (close to 0) -> large mismatch between distribution shapes

It captures precision-like behavior, and recall-like behavior.

### Human evaluations
Learning from human feedback - combined techniques:
- ADEM: a learned metric from human judgments for dialog system evaluation in a chatbot setting
- HUSE: Human Unified with Statistical Evaluation -> determines the similarity of the output distribution and a human reference distribution
