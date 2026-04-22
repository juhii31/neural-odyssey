"""
neural-odyssey | Module 02: Backpropagation by Hand
=====================================================
Module 01 taught us how data flows FORWARD through a network.
But the network didn't actually LEARN anything — we never updated weights.

Backpropagation is how learning happens.

It answers one question:
  "If I nudge weight W by a tiny amount, how much does the loss change?"

That answer is the GRADIENT. Once you have gradients, you update weights
in the direction that reduces loss. Do this thousands of times → learning.

What this module covers:
  1. The chain rule — the math behind backprop (one idea, not complicated)
  2. A complete forward + backward pass traced step-by-step on paper
  3. Gradient checking — verifying gradients numerically
  4. A full MLP trained with backprop, loss curve visualised
  5. Visualising gradient FLOW through layers (what dies, what lives)

Run this file:  python backprop.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# PART 1: THE CHAIN RULE — THE ONLY MATH YOU NEED
# =============================================================================
#
#  You want to know: how does loss L change when you change weight w?
#  Written as: dL/dw
#
#  The network is a chain of functions:
#    x → [Linear] → z → [Activation] → a → [Loss] → L
#
#  Chain rule says you can DECOMPOSE this:
#    dL/dw = dL/da × da/dz × dz/dw
#
#  Each term is LOCAL — easy to compute at each layer.
#  Multiply them all together and you get the full gradient.
#
#  BACKPROPAGATION is just applying the chain rule efficiently,
#  starting from the loss and working backwards layer by layer.
#
#  That's it. Every gradient descent paper, every transformer,
#  every LLM — all running backprop under the hood.
#
# =============================================================================


# =============================================================================
# PART 2: SCALAR EXAMPLE — TRACED BY HAND
# =============================================================================
#
#  Simplest possible network. One input, one weight, one output.
#
#  Forward:   z = w * x + b
#             a = sigmoid(z)
#             L = (a - y)^2        (MSE loss)
#
#  Backward (chain rule):
#             dL/da = 2*(a - y)
#             da/dz = sigmoid'(z) = a*(1-a)
#             dz/dw = x
#
#  Combined:  dL/dw = dL/da × da/dz × dz/dw
#                   = 2*(a-y) × a*(1-a) × x
#
# =============================================================================

def figure_01_scalar_backprop():
    """
    Visualise a single forward+backward pass as a computation graph.
    Shows each node, its value, and the gradient flowing backwards.
    """
    # Fixed values so we can trace exactly
    x = 2.0
    y = 1.0   # target
    w = 0.5
    b = 0.1

    # Forward pass
    z = w * x + b             # z = 1.1
    a = 1 / (1 + np.exp(-z))  # sigmoid(1.1) ≈ 0.75
    L = (a - y) ** 2          # loss ≈ 0.0625

    # Backward pass (chain rule, right to left)
    dL_da = 2 * (a - y)               # ∂L/∂a
    da_dz = a * (1 - a)               # ∂a/∂z  (sigmoid derivative)
    dz_dw = x                         # ∂z/∂w
    dz_db = 1.0                       # ∂z/∂b

    dL_dz = dL_da * da_dz             # ∂L/∂z  (chain rule)
    dL_dw = dL_dz * dz_dw             # ∂L/∂w
    dL_db = dL_dz * dz_db             # ∂L/∂b

    # ── Draw the computation graph ──
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    fig.patch.set_facecolor('#0F1117')
    ax.set_facecolor('#0F1117')

    nodes = [
        (1.5, 3.0, f'x={x}',        '#4A90D9', 'INPUT'),
        (1.5, 1.5, f'w={w}',        '#4A90D9', 'WEIGHT'),
        (1.5, 0.5, f'b={b}',        '#4A90D9', 'BIAS'),
        (4.5, 2.0, f'z={z:.3f}',    '#7B68EE', 'z=w·x+b'),
        (7.5, 2.0, f'a={a:.3f}',    '#50C878', 'σ(z)'),
        (10.5, 2.0,f'L={L:.4f}',   '#E24B4A', 'Loss'),
    ]

    for nx, ny, val, col, label in nodes:
        circle = plt.Circle((nx, ny), 0.55, color=col, zorder=3, alpha=0.85)
        ax.add_patch(circle)
        ax.text(nx, ny+0.05, val, ha='center', va='center',
                fontsize=8, color='white', fontweight='bold', zorder=4)
        ax.text(nx, ny-0.85, label, ha='center', va='center',
                fontsize=7, color='#AAAAAA', zorder=4)

    # Forward arrows (blue)
    arrows_fwd = [
        (2.05, 3.0, 1.9, 0),      # x → z
        (2.05, 1.5, 1.9, 0.4),    # w → z
        (2.05, 0.5, 1.9, 1.4),    # b → z
        (5.05, 2.0, 1.9, 0),      # z → a
        (8.05, 2.0, 1.9, 0),      # a → L
    ]
    for sx, sy, dx, dy in arrows_fwd:
        ax.annotate('', xy=(sx+dx, sy+dy), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle='->', color='#4A90D9', lw=1.5))

    # Backward gradient labels (orange, below arrows)
    grad_labels = [
        (6.0, 1.0, f'dL/dz={dL_dz:.4f}'),
        (9.0, 1.0, f'dL/da={dL_da:.4f}'),
    ]
    for gx, gy, glabel in grad_labels:
        ax.text(gx, gy, glabel, ha='center', fontsize=8,
                color='#FFA500', style='italic')

    ax.text(2.0, 4.5,
            f'dL/dw = dL/da × da/dz × dz/dw\n'
            f'      = {dL_da:.4f} × {da_dz:.4f} × {dz_dw:.1f}\n'
            f'      = {dL_dw:.4f}',
            fontsize=9, color='#FFA500',
            bbox=dict(boxstyle='round', facecolor='#1C1F2B', edgecolor='#FFA500', alpha=0.9))

    ax.set_title('Module 02 — Scalar Backpropagation: Computation Graph',
                 color='white', fontsize=12, pad=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'scalar_backprop.png')
    plt.savefig(path, dpi=130, bbox_inches='tight', facecolor='#0F1117')
    plt.close()
    print(f"  -> Saved: {path}")


# =============================================================================
# PART 3: GRADIENT CHECKING
# =============================================================================
#
#  How do you know your backprop is correct?
#  GRADIENT CHECK: compare analytical gradient vs numerical approximation.
#
#  Numerical gradient (finite difference):
#    df/dw ≈ [f(w + ε) - f(w - ε)] / (2ε)
#
#  If the analytical (backprop) gradient matches the numerical one
#  to ~5 decimal places, your backprop is correct.
#
#  Every serious ML engineer runs gradient checks when building new layers.
#
# =============================================================================

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_grad(a):
    """Gradient of sigmoid given its OUTPUT a (not input z)."""
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

def mse_loss(pred, target):
    return np.mean((pred - target) ** 2)


class LinearLayer:
    """A single fully-connected layer with complete forward + backward."""

    def __init__(self, in_dim, out_dim, activation='relu'):
        # He initialisation — correct scale for ReLU networks
        scale = np.sqrt(2.0 / in_dim)
        self.W = np.random.randn(out_dim, in_dim) * scale
        self.b = np.zeros((out_dim, 1))
        self.activation = activation

        # Cache for backprop
        self.x = None   # input to layer
        self.z = None   # pre-activation
        self.a = None   # post-activation

        # Gradients
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        Forward pass:
          z = W @ x + b       (linear transform)
          a = activation(z)   (non-linearity)
        """
        self.x = x
        self.z = self.W @ x + self.b

        if self.activation == 'relu':
            self.a = relu(self.z)
        elif self.activation == 'sigmoid':
            self.a = sigmoid(self.z)
        elif self.activation == 'linear':
            self.a = self.z
        return self.a

    def backward(self, dL_da):
        """
        Backward pass — chain rule applied to this layer.

        Receives: dL_da  — gradient of loss w.r.t. THIS layer's output
        Computes:
          dL_dz = dL_da × da/dz            (activation gradient)
          dL_dW = dL_dz @ x.T              (weight gradient)
          dL_db = sum(dL_dz)               (bias gradient)
          dL_dx = W.T @ dL_dz              (pass back to previous layer)

        Returns: dL_dx — gradient for the layer BEFORE this one
        """
        # Step 1: gradient through activation
        if self.activation == 'relu':
            dL_dz = dL_da * relu_grad(self.z)
        elif self.activation == 'sigmoid':
            dL_dz = dL_da * sigmoid_grad(self.a)
        elif self.activation == 'linear':
            dL_dz = dL_da

        # Step 2: gradients for weights and bias
        n = self.x.shape[1]
        self.dW = (dL_dz @ self.x.T) / n
        self.db = dL_dz.mean(axis=1, keepdims=True)

        # Step 3: gradient to pass to the previous layer
        dL_dx = self.W.T @ dL_dz
        return dL_dx


class MLP:
    """A multi-layer perceptron built from LinearLayer blocks."""

    def __init__(self, layer_dims, activations):
        assert len(activations) == len(layer_dims) - 1
        self.layers = [
            LinearLayer(layer_dims[i], layer_dims[i+1], activations[i])
            for i in range(len(activations))
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dL_dout):
        grad = dL_dout
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, lr):
        for layer in self.layers:
            if layer.dW is not None:
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db


def gradient_check(model, x, y_true, epsilon=1e-5):
    """
    Numerically verify analytical gradients via finite differences.
    Returns max relative error across all parameters.
    """
    # First do a forward + backward to get analytical gradients
    pred = model.forward(x)
    dL_dout = 2 * (pred - y_true) / y_true.size
    model.backward(dL_dout)

    errors = []

    for layer in model.layers:
        # Check W
        for i in range(layer.W.shape[0]):
            for j in range(layer.W.shape[1]):
                orig = layer.W[i, j]

                layer.W[i, j] = orig + epsilon
                loss_plus = mse_loss(model.forward(x), y_true)

                layer.W[i, j] = orig - epsilon
                loss_minus = mse_loss(model.forward(x), y_true)

                layer.W[i, j] = orig  # restore

                numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
                analytical_grad = layer.dW[i, j]

                rel_error = abs(numerical_grad - analytical_grad) / (
                    abs(numerical_grad) + abs(analytical_grad) + 1e-10)
                errors.append(rel_error)

    return max(errors), np.mean(errors)


def figure_02_gradient_check():
    """Run gradient check and visualise analytical vs numerical."""
    print("  Running gradient check...")

    # Small network for tractable check
    model = MLP([3, 8, 4, 1], ['relu', 'relu', 'sigmoid'])
    x = np.random.randn(3, 10)
    y = (np.random.randn(1, 10) > 0).astype(float)

    max_err, mean_err = gradient_check(model, x, y)

    print(f"  Max relative error:  {max_err:.2e}  {'✓ PASS' if max_err < 1e-4 else '✗ FAIL'}")
    print(f"  Mean relative error: {mean_err:.2e}")

    # Visualise: analytical vs numerical for each weight
    model2 = MLP([2, 6, 1], ['relu', 'sigmoid'])
    x2 = np.random.randn(2, 20)
    y2 = (np.random.randn(1, 20) > 0).astype(float)

    pred = model2.forward(x2)
    dL_dout = 2 * (pred - y2) / y2.size
    model2.backward(dL_dout)

    analytical, numerical = [], []
    eps = 1e-5

    for layer in model2.layers:
        for i in range(layer.W.shape[0]):
            for j in range(layer.W.shape[1]):
                orig = layer.W[i, j]
                layer.W[i, j] = orig + eps
                lp = mse_loss(model2.forward(x2), y2)
                layer.W[i, j] = orig - eps
                lm = mse_loss(model2.forward(x2), y2)
                layer.W[i, j] = orig
                numerical.append((lp - lm) / (2 * eps))
                analytical.append(layer.dW[i, j])

    analytical = np.array(analytical)
    numerical  = np.array(numerical)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#0F1117')

    # Scatter: analytical vs numerical (should be on y=x line)
    ax = axes[0]
    ax.set_facecolor('#1C1F2B')
    lim = max(abs(analytical).max(), abs(numerical).max()) * 1.1
    ax.scatter(numerical, analytical, s=20, alpha=0.7, color='#4A90D9')
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1.5, label='y = x (perfect)')
    ax.set_xlabel('Numerical gradient', color='white')
    ax.set_ylabel('Analytical gradient (backprop)', color='white')
    ax.set_title('Gradient Check\nAnalytical vs Numerical', color='white')
    ax.legend(facecolor='#1C1F2B', labelcolor='white', fontsize=8)
    ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#444')

    # Bar: relative errors per weight
    ax2 = axes[1]
    ax2.set_facecolor('#1C1F2B')
    rel_errors = np.abs(analytical - numerical) / (np.abs(analytical) + np.abs(numerical) + 1e-10)
    ax2.bar(range(len(rel_errors)), rel_errors, color='#50C878', alpha=0.8, width=1)
    ax2.axhline(1e-4, color='red', linestyle='--', lw=1.5, label='Threshold (1e-4)')
    ax2.set_xlabel('Weight index', color='white')
    ax2.set_ylabel('Relative error', color='white')
    ax2.set_title(f'Relative Error per Weight\nMax: {max_err:.2e}  Mean: {mean_err:.2e}',
                  color='white')
    ax2.legend(facecolor='#1C1F2B', labelcolor='white', fontsize=8)
    ax2.tick_params(colors='white')
    ax2.set_yscale('log')
    for sp in ax2.spines.values():
        sp.set_edgecolor('#444')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'gradient_check.png')
    plt.savefig(path, dpi=130, bbox_inches='tight', facecolor='#0F1117')
    plt.close()
    print(f"  -> Saved: {path}")


# =============================================================================
# PART 4: FULL TRAINING WITH LOSS CURVE
# =============================================================================

def figure_03_training_loss():
    """Train an MLP on a binary classification task, plot loss + accuracy."""
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=500, noise=0.25, random_state=42)
    X = X.T          # shape (2, 500)
    y = y.reshape(1, -1).astype(float)

    model = MLP([2, 16, 8, 1], ['relu', 'relu', 'sigmoid'])
    lr = 0.05
    epochs = 500

    losses, accs = [], []

    for epoch in range(epochs):
        # Forward
        pred = model.forward(X)

        # Binary cross-entropy loss
        eps = 1e-7
        loss = -np.mean(y * np.log(pred + eps) + (1 - y) * np.log(1 - pred + eps))

        # Backward
        dL_dout = (-(y / (pred + eps)) + (1 - y) / (1 - pred + eps)) / y.size
        model.backward(dL_dout)
        model.update(lr)

        # Track
        losses.append(loss)
        acc = ((pred > 0.5).astype(float) == y).mean()
        accs.append(acc)

        if epoch % 100 == 0:
            print(f"  Epoch {epoch:4d} | Loss: {loss:.4f} | Acc: {acc:.2%}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#0F1117')
    fig.suptitle('Module 02 — Full Training via Backpropagation', color='white', fontsize=13)

    # Loss curve
    ax = axes[0]
    ax.set_facecolor('#1C1F2B')
    ax.plot(losses, color='#E24B4A', lw=2)
    ax.set_xlabel('Epoch', color='white')
    ax.set_ylabel('Binary Cross-Entropy Loss', color='white')
    ax.set_title('Training Loss', color='white')
    ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#444')

    # Accuracy curve
    ax2 = axes[1]
    ax2.set_facecolor('#1C1F2B')
    ax2.plot(accs, color='#50C878', lw=2)
    ax2.set_ylim(0, 1.05)
    ax2.axhline(1.0, color='white', linestyle='--', lw=1, alpha=0.3)
    ax2.set_xlabel('Epoch', color='white')
    ax2.set_ylabel('Accuracy', color='white')
    ax2.set_title('Training Accuracy', color='white')
    ax2.tick_params(colors='white')
    for sp in ax2.spines.values():
        sp.set_edgecolor('#444')

    # Decision boundary
    ax3 = axes[2]
    ax3.set_facecolor('#1C1F2B')
    h = 0.02
    xx, yy = np.meshgrid(
        np.arange(X[0].min()-0.3, X[0].max()+0.3, h),
        np.arange(X[1].min()-0.3, X[1].max()+0.3, h)
    )
    grid = np.c_[xx.ravel(), yy.ravel()].T
    Z = model.forward(grid).reshape(xx.shape)
    ax3.contourf(xx, yy, Z, levels=50, cmap='RdBu_r', alpha=0.4)
    ax3.contour(xx, yy, Z, levels=[0.5], colors='white', linewidths=2)
    for c in range(2):
        mask = (y == c).ravel()
        ax3.scatter(X[0, mask], X[1, mask],
                    c=['#378ADD', '#E24B4A'][c], s=15, alpha=0.7)
    ax3.set_title(f'Decision Boundary\nFinal Acc: {accs[-1]:.2%}', color='white')
    ax3.tick_params(colors='white')
    for sp in ax3.spines.values():
        sp.set_edgecolor('#444')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'training_loss.png')
    plt.savefig(path, dpi=130, bbox_inches='tight', facecolor='#0F1117')
    plt.close()
    print(f"  -> Saved: {path}")


# =============================================================================
# PART 5: GRADIENT FLOW — WHAT DIES, WHAT LIVES
# =============================================================================
#
#  As gradients flow backwards, they get MULTIPLIED at each layer.
#  If any gradient < 1 repeatedly: product → 0   → VANISHING GRADIENT
#  If any gradient > 1 repeatedly: product → ∞   → EXPLODING GRADIENT
#
#  Sigmoid is notorious for vanishing gradients because:
#    σ'(z) = σ(z)(1-σ(z))  ≤ 0.25  always
#
#  ReLU avoids this because:
#    ReLU'(z) = 1 for z > 0  (gradient passes through unchanged)
#
# =============================================================================

def figure_04_gradient_flow():
    """Compare gradient magnitudes across layers for sigmoid vs relu."""
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=300, noise=0.2, random_state=0)
    X = X.T
    y = y.reshape(1, -1).astype(float)

    results = {}

    for activation, label in [('sigmoid', 'Sigmoid (vanishing)'), ('relu', 'ReLU (healthy)')]:
        dims = [2, 16, 16, 16, 16, 1]
        acts = [activation] * 4 + ['sigmoid']
        model = MLP(dims, acts)

        grad_norms = []
        for _ in range(5):   # collect over a few batches
            pred = model.forward(X)
            eps = 1e-7
            dL = (-(y / (pred + eps)) + (1 - y) / (1 - pred + eps)) / y.size
            model.backward(dL)

            norms = []
            for layer in model.layers:
                if layer.dW is not None:
                    norms.append(np.linalg.norm(layer.dW))
            grad_norms.append(norms)

        results[label] = np.mean(grad_norms, axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0F1117')
    ax.set_facecolor('#1C1F2B')

    colors = ['#E24B4A', '#50C878']
    for (label, norms), col in zip(results.items(), colors):
        ax.plot(range(1, len(norms)+1), norms, marker='o', lw=2.5,
                color=col, label=label, markersize=8)

    ax.set_xlabel('Layer (1 = closest to output, last = first layer)', color='white')
    ax.set_ylabel('||∇W||  (gradient norm)', color='white')
    ax.set_title('Gradient Flow: Sigmoid vs ReLU\n'
                 'Low norms in early layers = vanishing gradient = no learning',
                 color='white', fontsize=11)
    ax.legend(facecolor='#1C1F2B', labelcolor='white', fontsize=10)
    ax.tick_params(colors='white')
    ax.set_yscale('log')
    for sp in ax.spines.values():
        sp.set_edgecolor('#444')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'gradient_flow.png')
    plt.savefig(path, dpi=130, bbox_inches='tight', facecolor='#0F1117')
    plt.close()
    print(f"  -> Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print()
    print("=" * 60)
    print("  neural-odyssey | Module 02: Backpropagation")
    print("=" * 60)

    print("\n[ Scalar backprop computation graph ]")
    figure_01_scalar_backprop()

    print("\n[ Gradient check: analytical vs numerical ]")
    figure_02_gradient_check()

    print("\n[ Full training with loss curve ]")
    figure_03_training_loss()

    print("\n[ Gradient flow: sigmoid vs relu ]")
    figure_04_gradient_flow()

    print()
    print("=" * 60)
    print("  KEY TAKEAWAYS")
    print("=" * 60)
    print("""
  1. Backprop = chain rule applied layer by layer, right to left.
     Each layer computes LOCAL gradients; chain rule multiplies them.

  2. Gradient check is your sanity test. If analytical ≠ numerical
     (relative error > 1e-4), your math has a bug somewhere.

  3. Sigmoid kills gradients in deep nets. σ'(z) ≤ 0.25 always.
     Multiply that across 10 layers → gradient ≈ 0 → no learning.

  4. ReLU passes gradients through unchanged for positive inputs.
     That's why deep networks train at all.

  5. This is the algorithm behind EVERY neural network ever trained.
     GPT, BERT, ResNet — all backprop under the hood.
    """)
    print("=" * 60)
    print()
    print("  Output PNGs saved to: module_02_backprop/output/")
    print("  ├── scalar_backprop.png  — computation graph traced")
    print("  ├── gradient_check.png  — analytical vs numerical")
    print("  ├── training_loss.png   — full training run")
    print("  └── gradient_flow.png   — vanishing gradient shown")
    print()
    print("  Commit: git add . && git commit -m 'Module 02: Backprop complete'")
    print()
