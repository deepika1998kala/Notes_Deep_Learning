Of course! Here’s the **Canvas-style notes** I had earlier provided for **Deep Learning Activation Functions** — I'll recreate them neatly for you:

---

# 🎯 Deep Learning — Activation Functions (Quick Canvas Notes)

---

## 1. **What are Activation Functions?**
- Introduce **non-linearity** into the model.
- Help the neural network **learn complex patterns**.
- Decide whether a neuron should be **activated** or **not** (based on input).

---

## 2. **Types of Activation Functions**

### 🔵 2.1 Linear Activation
- Formula: `f(x) = x`
- Problem: No non-linearity → Model behaves like a linear regression.

---

### 🟢 2.2 Sigmoid Function
- Formula: `f(x) = 1 / (1 + e^(-x))`
- Output: (0, 1)
- **Pros**: Smooth curve, probabilistic interpretation.
- **Cons**: 
  - **Vanishing Gradient Problem**: Gradients become very small for large |x|.
  - Slow convergence.

---

### 🟡 2.3 Tanh (Hyperbolic Tangent)
- Formula: `f(x) = (e^x - e^-x) / (e^x + e^-x)`
- Output: (-1, 1)
- **Pros**: Centered around 0 (better than sigmoid).
- **Cons**: Still suffers from vanishing gradients for large values.

---

### 🔴 2.4 ReLU (Rectified Linear Unit)
- Formula: `f(x) = max(0, x)`
- **Pros**: 
  - Very efficient computation.
  - Reduces vanishing gradient problem.
- **Cons**: 
  - **Dying ReLU problem**: Some neurons may become inactive permanently if outputs become 0.

---

### 🟠 2.5 Leaky ReLU
- Formula: `f(x) = x if x > 0 else 0.01x`
- **Pros**: Fixes dying ReLU by allowing a small, non-zero gradient when x < 0.

---

### 🟣 2.6 Parametric ReLU (PReLU)
- Like Leaky ReLU, but slope (α) is **learned during training**.
- Formula: `f(x) = x if x > 0 else αx`

---

### 🟤 2.7 ELU (Exponential Linear Unit)
- Formula:
  - `f(x) = x` for x > 0
  - `f(x) = α*(e^x - 1)` for x ≤ 0
- **Pros**: 
  - Avoids dying ReLU.
  - Smooth curve.
- **Cons**: Slightly slower than ReLU.

---

### ⚫ 2.8 Swish (by Google)
- Formula: `f(x) = x * sigmoid(x)`
- **Pros**: 
  - Smooth, non-monotonic.
  - Performs better than ReLU in some models.

---

## 3. **Activation Function Selection Tips**
| Task | Common Activation |
|:----|:-------------------|
| Hidden Layers | ReLU / Leaky ReLU |
| Output (Binary Classification) | Sigmoid |
| Output (Multi-class Classification) | Softmax |
| Output (Regression) | Linear |

---

## 4. **Problems in Activation Functions**
- **Vanishing Gradient**: Gradients become too small, slow learning.
- **Exploding Gradient**: Gradients become too large, unstable learning.
- **Dead Neurons**: Especially in ReLU if outputs become permanently zero.

---

# ⚡ Summary Diagram
---
