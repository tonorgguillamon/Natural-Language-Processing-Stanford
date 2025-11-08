## 🧩 1. Embeddings

- **Definition:**  
  `nn.Embedding(num_embeddings, embedding_dim)` creates a lookup table mapping integer indices (e.g., token IDs) to dense, trainable vectors.

- Conceptually, this is just a matrix of shape `(num_embeddings, embedding_dim)` where:
  - Each row corresponds to a word or token.
  - Each column represents a dimension in the embedding space.

- **Example:**
  ```python
  embedding = nn.Embedding(10, 3)
  input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
  embedding(input)
  ```

- This performs a **lookup** of each integer and returns its learned embedding vector.

- The embedding layer’s weights are learned during training, just like any other parameters.

---

## ⚙️ 2. Model Structure and `nn.Module`

- In PyTorch, all models subclass `nn.Module`.  
  You define:
  - Layers in `__init__`
  - The forward pass logic in `forward()`

- **Example:**
  ```python
  class MyModel(nn.Module):
      def __init__(self):
          super().__init__()
          self.layer = nn.Linear(10, 5)
      
      def forward(self, x):
          return F.relu(self.layer(x))
  ```

- `nn.Linear`, `nn.Dropout`, etc., are callable because they implement `__call__()` internally.  
  You can treat them like functions: `output = self.layer(x)`.

---

## 🔢 3. Tensor `.view()` Method

- Used to **reshape** tensors without copying memory.
- Example:
  ```python
  x = torch.randn(2, 3, 4)
  x = x.view(2, -1)  # shape becomes (2, 12)
  ```
- The `-1` automatically infers the correct dimension size.

---

## 🧮 4. Training a Model — Core Steps

Every training loop involves the same key steps:

1. **Forward pass** — compute predictions  
2. **Compute loss** — compare predictions vs true labels  
3. **Zero gradients** — clear old gradient values (`optimizer.zero_grad()`)  
4. **Backward pass** — compute new gradients (`loss.backward()`)  
5. **Update parameters** — adjust model weights (`optimizer.step()`)

---

## 📉 5. Gradients


- The **gradient** is the derivative of the loss with respect to a parameter:

$$
\frac{\partial L}{\partial \theta}
$$


- It shows how much and in what direction the loss changes if you slightly adjust that parameter.

- The gradient points in the direction of **steepest increase** of loss.  
  Therefore, we move in the **opposite direction** to minimize the loss:

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \nabla_\theta L(\theta)
$$

where $\eta$ = learning rate.

---

## ⚖️ 6. Bias Term

In a linear transformation:

$$
y = W x + b
$$

 - $b$ is the **bias**, which shifts the output even if all inputs are zero.  
  This allows the model to fit data not passing through the origin.

---

## 🧭 7. Adam Optimizer

Adam = **Adaptive Moment Estimation**

It combines **momentum** and **adaptive learning rates** per parameter.

| Term | Meaning | Typical Value |
|------|----------|----------------|
| β₁ | decay rate for mean of gradients | 0.9 |
| β₂ | decay rate for variance of gradients | 0.999 |

### Update Rule:
$$
	heta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

- **β₁ (momentum)** controls how much past gradients affect the current mean.
  - High β₁ (≈ 0.9): smooths updates, more stable.
  - Low β₁: reacts faster but can be noisy.

### Flat Loss Surfaces:
- Parameters with smaller gradients (flatter regions) get relatively larger updates → helps escape plateaus.

---

## 💧 8. Dropout

### What it does:
- Randomly “drops” (sets to zero) some activations during **training**.
- Prevents overfitting by discouraging the network from relying on specific neurons.

### Why not during testing:
- At **test time**, we want stable, deterministic predictions.
- Dropout is turned off with `model.eval()`.

### Scaling:
- During training, active neurons are scaled by `1 / (1 - dropout_rate)` to maintain consistent expectations.

```python
model.train()  # enables dropout
model.eval()   # disables dropout
```

---

## 🔢 9. Xavier (Glorot) Initialization

Xavier initialization keeps the variance of activations consistent across layers, preventing gradients from exploding or vanishing.

$$
\text{Var}(W) = \frac{2}{n_{in} + n_{out}}
$$

- Ensures signals and gradients neither shrink nor grow as they pass through layers.

Implemented in PyTorch:
```python
nn.init.xavier_uniform_(layer.weight)
```

---

## 🧠 10. Key Concept Summary

| Concept | Meaning | Formula / Example |
|----------|----------|------------------|
| **Embedding lookup** | Convert token indices to vectors | `embedding(input)` |
| **Gradient** | Direction of loss increase | ∇L(θ) |
| **Optimizer step** | Move opposite gradient | θ ← θ − η∇L |
| **Adam optimizer** | Adaptive + momentum-based | uses β₁, β₂ |
| **Bias** | Adds offset | y = Wx + b |
| **Dropout** | Randomly disables neurons | Regularization |
| **`train()` / `eval()`** | Toggles behavior | — |
| **`view()`** | Reshape tensor | `x.view(batch, -1)` |
| **Xavier Init** | Keeps variance stable | 2 / (n_in + n_out) |

---

## 🧩 11. Full Model Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MyModel(nn.Module):
    def __init__(self, embedding_matrix, num_features=36, hidden_dim=200, num_classes=3, dropout_rate=0.5):
        super().__init__()
        torch.manual_seed(0)

        self.embedding_dim = embedding_matrix.shape[1]

        # Embedding layer
        self.embedding_layer = nn.Embedding(embedding_matrix.shape[0], self.embedding_dim)
        self.embedding_layer.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        # Linear → Hidden
        self.fc_in = nn.Linear(num_features * self.embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

        # Xavier initialization
        nn.init.xavier_uniform_(self.fc_in.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def lookup_embeddings(self, token_tensor):
        x = self.embedding_layer(token_tensor)
        x = x.view(x.shape[0], -1)
        return x

    def forward(self, tokens):
        x = self.lookup_embeddings(tokens)
        x = F.relu(self.fc_in(x))
        x = self.dropout(x)
        logits = self.fc_out(x)
        return logits
```

---

## 🧩 12. Training Loop

```python
def train_model(model, data_loader, num_epochs=5, learning_rate=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for inputs, labels in data_loader:
            optimizer.zero_grad()       # 1. reset gradients
            outputs = model(inputs)     # 2. forward pass
            loss = criterion(outputs, labels)
            loss.backward()             # 3. backpropagation
            optimizer.step()            # 4. update weights
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {total_loss:.4f}")

    print("✅ Training complete.")
```

---

## 🧭 13. End-to-End Training Flow

1. Input tokens → indices → **Embeddings**  
2. Embeddings → **Linear transformation** → hidden layer  
3. Apply **ReLU** non-linearity  
4. Apply **Dropout** (only during training)  
5. Hidden → **Output logits**  
6. Compare logits vs true labels → **Loss**  
7. **Backward pass** (compute gradients)  
8. **Optimizer step** (update weights)  
9. Repeat for all batches

---

## ✅ 14. Quick Summary Table

| Step | Concept | Description |
|------|----------|-------------|
| 1️⃣ | **Embedding lookup** | Map indices to dense vectors |
| 2️⃣ | **Linear layer** | Weighted transformation |
| 3️⃣ | **ReLU** | Add non-linearity |
| 4️⃣ | **Dropout** | Regularization |
| 5️⃣ | **Loss** | Compare predictions to true labels |
| 6️⃣ | **Backpropagation** | Compute gradients |
| 7️⃣ | **Optimizer (Adam)** | Update weights adaptively |
| 8️⃣ | **Repeat** | Until convergence |

---

## 📚 References

- [PyTorch Embedding Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- [An Explanation of Xavier Initialization](http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---
