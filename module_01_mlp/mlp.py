"""
neural-odyssey | Module 01: MLP + Forward Pass
===============================================
Module 00 taught us the perceptron has one fatal flaw: it draws one straight
line. XOR broke it.

The fix: stack multiple perceptrons into LAYERS.
This is the Multi-Layer Perceptron (MLP) — the backbone of all deep learning.

What this module covers:
  1. Why depth matters — from lines to curves
  2. Activation functions — what they are and why we need them
  3. The forward pass — exactly how data flows through layers
  4. Comparing activations: Step vs Sigmoid vs Tanh vs ReLU
  5. Solving XOR with a 2-layer MLP (what Module 00 couldn't do)
  6. Visualising how each hidden layer transforms the data

Run this file:  python mlp.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

np.random.seed(42)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# PART 1: WHY DO WE NEED ACTIVATION FUNCTIONS?
# =============================================================================
#
#  A layer computes:  output = activation(W @ x + b)
#
#  What if we use NO activation (just linear)?
#    Layer 1: y₁ = W₁x + b₁
#    Layer 2: y₂ = W₂y₁ + b₂ = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁+b₂)
#
#  Two linear layers collapse into ONE linear layer.
#  No matter how many layers you stack — still just one straight line.
#  You gain NOTHING from depth without a non-linear activation.
#
#  Activation functions introduce non-linearity — they let the network
#  learn CURVES, not just lines.
#
# =============================================================================

def step(z):
    """The original perceptron activation. Binary — 0 or 1. Not differentiable."""
    return (z >= 0).astype(float)

def sigmoid(z):
    """
    Squashes any value to (0, 1). Smooth version of step.
    Problem: for large |z|, gradient ≈ 0 → 'vanishing gradient' in deep nets.
    σ(z) = 1 / (1 + e^(-z))
    """
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def tanh(z):
    """
    Squashes to (-1, 1). Zero-centred (better than sigmoid for training).
    Still suffers vanishing gradient for large |z|.
    """
    return np.tanh(z)

def relu(z):
    """
    Rectified Linear Unit: max(0, z).
    Simple, fast, no vanishing gradient for positive values.
    Default choice for hidden layers in modern networks.
    """
    return np.maximum(0, z)

def relu_grad(z):
    """Gradient of ReLU: 1 if z > 0, else 0."""
    return (z > 0).astype(float)

def softmax(z):
    """
    Converts raw scores → probabilities that sum to 1.
    Used in the OUTPUT layer for classification.
    """
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# =============================================================================
# PART 2: A SINGLE DENSE LAYER
# =============================================================================
#
#  Each layer holds:
#    W — weight matrix  shape: (input_size, output_size)
#    b — bias vector    shape: (1, output_size)
#
#  Forward pass (for a batch of inputs X, shape: (batch, input_size)):
#    z = X @ W + b      ← weighted sum   (batch, output_size)
#    a = activation(z)  ← apply non-linearity
#
#  Think of W as: "how much does each input neuron influence each output neuron?"
#  Every input connects to every output — that's why it's called 'fully connected'.
#
# =============================================================================

class Layer:
    def __init__(self, input_size, output_size, activation='relu'):
        # He initialisation: keeps signal strength stable through many layers
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((1, output_size))
        self.activation = activation

        # Filled during backward pass (Module 02)
        self.dW = None
        self.db = None
        self._X = None
        self._z = None

    def forward(self, X):
        """
        X shape: (batch_size, input_size)
        Returns activated output shape: (batch_size, output_size)
        """
        self._X = X
        self._z = X @ self.W + self.b   # core computation — a matrix multiply

        if self.activation == 'relu':     return relu(self._z)
        if self.activation == 'sigmoid':  return sigmoid(self._z)
        if self.activation == 'tanh':     return tanh(self._z)
        if self.activation == 'softmax':  return softmax(self._z)
        if self.activation == 'linear':   return self._z
        raise ValueError(f"Unknown activation: {self.activation}")

    def backward(self, dA):
        """Backprop — covered in detail in Module 02."""
        n = self._X.shape[0]
        if self.activation == 'relu':
            dZ = dA * relu_grad(self._z)
        else:
            dZ = dA
        self.dW = (self._X.T @ dZ) / n
        self.db = dZ.mean(axis=0, keepdims=True)
        return dZ @ self.W.T


# =============================================================================
# PART 3: THE MLP — STACKED LAYERS
# =============================================================================
#
#  The key idea: each hidden layer TRANSFORMS the data into a new space
#  where the problem becomes easier to solve.
#
#  In XOR:
#    Input space:  the 4 points form an X — not linearly separable
#    Hidden space: the first layer WARPS the space so a line CAN separate them
#    Output layer: draws that separating line
#
#  This is what "representation learning" means — the network learns
#  a better way to represent the data, not just a classifier on top of it.
#
# =============================================================================

class MLP:
    def __init__(self, layer_sizes, activations):
        """
        layer_sizes : e.g. [2, 4, 1]  →  2 inputs, 4 hidden, 1 output
        activations : e.g. ['relu', 'sigmoid']  →  one per layer transition
        """
        assert len(activations) == len(layer_sizes) - 1
        self.layers = [
            Layer(layer_sizes[i], layer_sizes[i+1], activations[i])
            for i in range(len(activations))
        ]

    def forward(self, X):
        """Run X through all layers. Returns final output."""
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def forward_with_intermediates(self, X):
        """
        Same as forward, but also returns the output of EVERY layer.
        This lets us visualise what each hidden layer 'sees'.
        """
        activations = [X]
        out = X
        for layer in self.layers:
            out = layer.forward(out)
            activations.append(out)
        return activations

    def backward(self, dLoss):
        grad = dLoss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def predict(self, X):
        out = self.forward(X)
        if out.shape[1] == 1:
            return (out >= 0.5).astype(int).flatten()
        return np.argmax(out, axis=1)


# =============================================================================
# PART 4: TRAINING (same structure as Module 00, but now with gradient descent)
# =============================================================================

def binary_cross_entropy(probs, y):
    """Loss for binary classification (one output neuron, sigmoid)."""
    p = np.clip(probs.flatten(), 1e-12, 1 - 1e-12)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    grad = ((p - y) / (p * (1 - p))) / len(y)
    return loss, grad.reshape(-1, 1)

def train(model, X, y, epochs=5000, lr=0.1):
    loss_history = []
    for epoch in range(epochs):
        probs = model.forward(X)
        loss, grad = binary_cross_entropy(probs, y)
        model.backward(grad)
        for layer in model.layers:
            if layer.dW is not None:
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db
        loss_history.append(loss)
        if epoch % 1000 == 0:
            acc = (model.predict(X) == y).mean()
            print(f"  Epoch {epoch:5d} | Loss: {loss:.4f} | Accuracy: {acc:.0%}")
    return loss_history


# =============================================================================
# VISUALISATIONS
# =============================================================================

def figure_01_activation_functions():
    """
    Figure 1: Compare all activation functions side by side.
    This is the most referenced figure in all of deep learning education.
    Know these shapes — you will see them everywhere.
    """
    z = np.linspace(-6, 6, 300)
    funcs = {
        'Step (perceptron)': (step(z),    '#888780', '--'),
        'Sigmoid':           (sigmoid(z), '#534AB7', '-'),
        'Tanh':              (tanh(z),    '#1D9E75', '-'),
        'ReLU':              (relu(z),    '#E24B4A', '-'),
    }

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    fig.suptitle('Activation functions — the non-linearity that makes depth work',
                 fontsize=13, y=1.02)

    notes = [
        'Binary. Not\ndifferentiable.\nCan\'t use\ngradient descent.',
        'Smooth 0→1.\nUsed in output\nfor binary class.\nVanishes for large z.',
        'Smooth -1→1.\nZero-centred.\nBetter than sigmoid\nfor hidden layers.',
        'max(0,z).\nFast. No vanish\nfor positive z.\nDefault choice today.',
    ]

    for ax, (name, (y_vals, color, ls)), note in zip(axes, funcs.items(), notes):
        ax.plot(z, y_vals, color=color, linewidth=2.5, linestyle=ls)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('z  (input)', fontsize=9)
        ax.set_ylabel('output', fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(-6, 6)
        ax.text(0.05, 0.05, note, transform=ax.transAxes,
                fontsize=8, color='#555', verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'activation_functions.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {path}")


def figure_02_xor_solved():
    """
    Figure 2: MLP solving XOR — what the perceptron couldn't do.
    We also visualise the HIDDEN LAYER activations to show how
    the network transforms the input space.
    """
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0, 1, 1, 0], dtype=float)

    # Architecture: 2 inputs → 4 hidden (relu) → 1 output (sigmoid)
    model = MLP([2, 4, 1], ['relu', 'sigmoid'])
    print("\n  Training MLP on XOR:")
    loss_history = train(model, X, y, epochs=5000, lr=0.1)

    preds = model.predict(X)
    acc = (preds == y).mean()
    print(f"  Final accuracy: {acc:.0%}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('MLP solves XOR — what the perceptron could not', fontsize=13, y=1.02)

    # --- Left: decision boundary ---
    ax = axes[0]
    h = 0.01
    xx, yy = np.meshgrid(np.arange(-0.5, 1.5, h), np.arange(-0.5, 1.5, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.forward(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=50, cmap='RdBu_r', alpha=0.4)
    ax.contour(xx, yy, Z, levels=[0.5], colors='#534AB7', linewidths=2)
    colors = ['#378ADD', '#E24B4A']
    for c in range(2):
        mask = y == c
        ax.scatter(X[mask,0], X[mask,1], c=colors[c], s=250, zorder=5,
                   edgecolors='white', linewidth=1.5, label=f'XOR={c}')
    ax.set_title('Decision boundary\n(curved — not possible with 1 perceptron)', fontsize=10)
    ax.legend(); ax.grid(True, alpha=0.2); ax.set_aspect('equal')
    ax.set_xlabel('x₁'); ax.set_ylabel('x₂')

    # --- Middle: hidden layer activations ---
    ax2 = axes[1]
    intermediates = model.forward_with_intermediates(X)
    hidden = intermediates[1]   # shape: (4, 4) — 4 data points, 4 hidden neurons

    # Plot data in the hidden space (first 2 hidden neurons as axes)
    for c in range(2):
        mask = y == c
        ax2.scatter(hidden[mask, 0], hidden[mask, 1],
                    c=colors[c], s=250, zorder=5,
                    edgecolors='white', linewidth=1.5, label=f'XOR={c}')
        for i, xi in enumerate(X[mask]):
            ax2.annotate(f'({int(xi[0])},{int(xi[1])})',
                         (hidden[mask][i,0], hidden[mask][i,1]),
                         fontsize=9, textcoords='offset points', xytext=(5,5))
    ax2.set_title('Hidden layer space\n(data is now linearly separable!)', fontsize=10)
    ax2.set_xlabel('Hidden neuron 1 output')
    ax2.set_ylabel('Hidden neuron 2 output')
    ax2.legend(); ax2.grid(True, alpha=0.2)

    # --- Right: training loss ---
    ax3 = axes[2]
    ax3.plot(loss_history, color='#534AB7', linewidth=1.5)
    ax3.set_title('Training loss\n(converges — unlike the perceptron)', fontsize=10)
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('Loss')
    ax3.grid(True, alpha=0.2); ax3.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'xor_solved_mlp.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {path}")


def figure_03_forward_pass_trace():
    """
    Figure 3: Trace the EXACT numbers through a tiny MLP by hand.
    This makes the forward pass completely concrete — no black boxes.
    """
    print("\n  Hand-tracing forward pass through a tiny MLP...")

    # Tiny fixed network so we can trace every number
    np.random.seed(0)
    model = MLP([2, 3, 2], ['relu', 'softmax'])

    x = np.array([[0.5, 0.8]])   # one input sample

    # Layer 1
    z1 = x @ model.layers[0].W + model.layers[0].b
    a1 = relu(z1)

    # Layer 2
    z2 = a1 @ model.layers[1].W + model.layers[1].b
    a2 = softmax(z2)

    print(f"\n  Input x         : {x.flatten()}")
    print(f"  Layer 1 weights W1 shape: {model.layers[0].W.shape}")
    print(f"  z1 = x @ W1 + b1: {z1.flatten().round(4)}")
    print(f"  a1 = relu(z1)   : {a1.flatten().round(4)}")
    print(f"  Layer 2 weights W2 shape: {model.layers[1].W.shape}")
    print(f"  z2 = a1 @ W2 + b2: {z2.flatten().round(4)}")
    print(f"  a2 = softmax(z2): {a2.flatten().round(4)}")
    print(f"  Predicted class : {np.argmax(a2)}")
    print(f"  Sum of probs    : {a2.sum():.6f}  (always = 1.0)")

    # Visualise the flow
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axis('off')
    fig.patch.set_facecolor('#fafafa')

    def draw_box(ax, x, y, w, h, text, color, fontsize=9):
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.02",
                             facecolor=color, edgecolor='white',
                             linewidth=1.5, zorder=3)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, zorder=4,
                color='white' if color not in ['#f0f0f0'] else '#333')

    # Input
    draw_box(ax, 0.02, 0.35, 0.12, 0.3, f'Input\nx₁={x[0,0]}\nx₂={x[0,1]}', '#378ADD')
    # Arrow
    ax.annotate('', xy=(0.17, 0.5), xytext=(0.14, 0.5),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))
    # z1
    z1_str = '\n'.join([f'z₁={v:.3f}' for v in z1.flatten()])
    draw_box(ax, 0.17, 0.25, 0.13, 0.5, f'z = x@W+b\n{z1_str}', '#7F77DD')
    ax.annotate('', xy=(0.33, 0.5), xytext=(0.30, 0.5),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))
    # a1
    a1_str = '\n'.join([f'a={v:.3f}' for v in a1.flatten()])
    draw_box(ax, 0.33, 0.25, 0.13, 0.5, f'ReLU(z)\n{a1_str}', '#1D9E75')
    ax.annotate('', xy=(0.49, 0.5), xytext=(0.46, 0.5),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))
    # z2
    z2_str = '\n'.join([f'z₂={v:.3f}' for v in z2.flatten()])
    draw_box(ax, 0.49, 0.3, 0.13, 0.4, f'z = a@W+b\n{z2_str}', '#7F77DD')
    ax.annotate('', xy=(0.65, 0.5), xytext=(0.62, 0.5),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))
    # a2
    a2_str = '\n'.join([f'p={v:.3f}' for v in a2.flatten()])
    draw_box(ax, 0.65, 0.3, 0.14, 0.4, f'Softmax(z)\n{a2_str}', '#E24B4A')
    ax.annotate('', xy=(0.82, 0.5), xytext=(0.79, 0.5),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))
    # Output
    draw_box(ax, 0.82, 0.38, 0.12, 0.24,
             f'Class {np.argmax(a2)}\n(p={a2.max():.3f})', '#E8593C')

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title('Forward pass — exact numbers flowing through a [2→3→2] MLP',
                 fontsize=12, pad=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'forward_pass_trace.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {path}")


def figure_04_depth_matters():
    """
    Figure 4: Train networks of different depths on the spiral dataset.
    Deeper networks learn more complex decision boundaries.
    """
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=400, noise=0.2, random_state=42)
    y = y.astype(float)

    configs = [
        ([2, 1],       ['sigmoid'],                  '0 hidden layers\n(just a perceptron)'),
        ([2, 4, 1],    ['relu', 'sigmoid'],           '1 hidden layer\n(4 neurons)'),
        ([2,16,16,1],  ['relu','relu','sigmoid'],     '2 hidden layers\n(16 neurons each)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Depth matters — more layers = more complex boundaries', fontsize=13, y=1.02)

    for ax, (sizes, acts, title) in zip(axes, configs):
        model = MLP(sizes, acts)
        # Quick train
        for _ in range(3000):
            probs = model.forward(X)
            loss, grad = binary_cross_entropy(probs, y)
            model.backward(grad)
            for layer in model.layers:
                if layer.dW is not None:
                    layer.W -= 0.05 * layer.dW
                    layer.b -= 0.05 * layer.db

        acc = (model.predict(X) == y).mean()

        h = 0.02
        xx, yy = np.meshgrid(np.arange(X[:,0].min()-0.3, X[:,0].max()+0.3, h),
                              np.arange(X[:,1].min()-0.3, X[:,1].max()+0.3, h))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model.forward(grid).reshape(xx.shape)

        ax.contourf(xx, yy, Z, levels=50, cmap='RdBu_r', alpha=0.35)
        ax.contour(xx, yy, Z, levels=[0.5], colors='#534AB7', linewidths=2)
        for c in range(2):
            mask = y == c
            ax.scatter(X[mask,0], X[mask,1],
                       c=['#378ADD','#E24B4A'][c], s=20, alpha=0.7)
        n_params = sum(l.W.size + l.b.size for l in model.layers)
        ax.set_title(f'{title}\nAccuracy: {acc:.0%} | Params: {n_params}', fontsize=10)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'depth_matters.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print()
    print("=" * 60)
    print("  neural-odyssey | Module 01: MLP + Forward Pass")
    print("=" * 60)

    print("\n[ Activation functions ]")
    figure_01_activation_functions()

    print("\n[ XOR solved by MLP ]")
    figure_02_xor_solved()

    print("\n[ Forward pass trace ]")
    figure_03_forward_pass_trace()

    print("\n[ Depth comparison ]")
    figure_04_depth_matters()

    print()
    print("=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print("""
  KEY TAKEAWAYS

  1. Without activation functions, stacking layers is useless.
     N linear layers = 1 linear layer. You need non-linearity.

  2. ReLU is the default for hidden layers today.
     Fast, simple, no vanishing gradient for positive inputs.

  3. Softmax goes on the OUTPUT layer for classification.
     It turns raw scores into probabilities that sum to 1.

  4. The forward pass is just repeated matrix multiply + activation.
     output = activation(X @ W + b), layer by layer.

  5. Hidden layers TRANSFORM the data into a new space
     where the problem is easier to solve (XOR example).

  6. More depth = more complex boundaries, but also more
     parameters to train. Tradeoff covered in Module 04.

  Output files saved to: module_01_mlp/output/
  Next: Module 02 — Backpropagation (how the network actually learns)
    """)
    print("=" * 60)
