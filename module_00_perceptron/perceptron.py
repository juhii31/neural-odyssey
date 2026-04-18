"""
neural-odyssey | Module 00: The Perceptron
==========================================
Year: 1957. Inventor: Frank Rosenblatt.
Hardware: a room-sized machine called the Mark I.
Goal: teach a machine to recognise images.

The perceptron is the simplest possible "neural network" — a single neuron.
Understanding it deeply means you understand the atom of all deep learning.

What this module covers:
  1. What a perceptron computes (the math)
  2. How it learns (the perceptron learning rule)
  3. What it can learn (AND, OR gates)
  4. What it CAN'T learn (XOR) — and why that matters enormously
  5. Visualising the decision boundary as it learns, step by step

Run this file:  python perceptron.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

np.random.seed(42)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# PART 1: WHAT DOES A PERCEPTRON COMPUTE?
# =============================================================================
#
#  Inputs: x₁, x₂, ..., xₙ  (numbers — could be pixels, prices, anything)
#  Weights: w₁, w₂, ..., wₙ (how much each input matters)
#  Bias: b                   (shifts the decision threshold)
#
#  Step 1 — weighted sum:  z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
#  Step 2 — decision:      output = 1 if z >= 0, else 0
#
#  The output is binary — yes or no, fire or don't fire.
#  This is called a "step function" activation.
#
#  Geometrically: the weights define a LINE (in 2D), PLANE (in 3D),
#  or HYPERPLANE (in higher dimensions). Everything on one side = class 1,
#  everything on the other = class 0. The perceptron draws a straight line.
#
# =============================================================================

class Perceptron:
    def __init__(self, n_inputs, lr=0.1):
        """
        n_inputs: how many features each data point has
        lr:       learning rate — how big a step we take when we're wrong

        Weights start near zero. We'll learn their values from data.
        """
        self.w = np.zeros(n_inputs)   # one weight per input feature
        self.b = 0.0                  # bias — a learnable offset
        self.lr = lr

        # We'll record every weight update so we can animate learning
        self.history = []             # list of (w, b) snapshots

    def predict(self, X):
        """
        X: input matrix, shape (n_samples, n_features)
        Returns: binary predictions, shape (n_samples,)

        The step function: fire if the weighted sum >= 0.
        """
        z = X @ self.w + self.b       # weighted sum for all samples at once
        return (z >= 0).astype(int)   # 1 or 0

    def train(self, X, y, epochs=10):
        """
        The Perceptron Learning Rule — remarkably simple:

          For each training example (x, y_true):
            1. Predict: y_hat = step(w·x + b)
            2. Compute error: error = y_true - y_hat
               → if correct: error = 0, do nothing
               → if predicted 0, truth is 1: error = +1, push weights UP
               → if predicted 1, truth is 0: error = -1, push weights DOWN
            3. Update: w = w + lr × error × x
                       b = b + lr × error

        This is NOT gradient descent (no loss function, no calculus).
        It's an older, simpler rule — but it PROVABLY converges
        if the data is linearly separable.
        """
        self.history = [(self.w.copy(), self.b)]  # snapshot before training

        for epoch in range(epochs):
            n_errors = 0
            for xi, yi in zip(X, y):
                y_hat = self.predict(xi.reshape(1, -1))[0]
                error = yi - y_hat
                if error != 0:
                    n_errors += 1
                    self.w += self.lr * error * xi
                    self.b += self.lr * error
                    self.history.append((self.w.copy(), self.b))

            acc = (self.predict(X) == y).mean()
            print(f"  Epoch {epoch+1:2d} | Errors: {n_errors} | Accuracy: {acc:.0%}")
            if n_errors == 0:
                print(f"  [OK] Converged at epoch {epoch+1}")
                break

        return self


# =============================================================================
# PART 2: THE DATASETS — LOGIC GATES
# =============================================================================
#
#  Logic gates are the perfect test: they're tiny, exact, and we know
#  the ground truth. Every input is a pair of binary values (0 or 1).
#
#  AND gate:  output 1 only if BOTH inputs are 1
#  OR gate:   output 1 if EITHER input is 1
#  XOR gate:  output 1 if inputs are DIFFERENT — the famous impossible case
#
#  AND and OR are linearly separable — a straight line can split them.
#  XOR is NOT — no straight line can ever separate it.
#  This limitation, proved in 1969, nearly killed neural network research.
#

def make_logic_gate(gate_type):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    if gate_type == 'AND':
        y = np.array([0, 0, 0, 1])
    elif gate_type == 'OR':
        y = np.array([0, 1, 1, 1])
    elif gate_type == 'XOR':
        y = np.array([0, 1, 1, 0])
    else:
        raise ValueError(f"Unknown gate: {gate_type}")
    return X, y


# =============================================================================
# PART 3: VISUALISATION HELPERS
# =============================================================================

def plot_decision_boundary(ax, w, b, X, y, title, show_line=True):
    """
    Draw the data points and the perceptron's current decision boundary.

    The boundary is the line where z = w·x + b = 0.
    In 2D with inputs x₁, x₂:
        w₀·x₁ + w₁·x₂ + b = 0
        → x₂ = -(w₀·x₁ + b) / w₁

    Points on one side of this line → predict 1.
    Points on the other side → predict 0.
    """
    colors = ['#378ADD', '#E24B4A']   # blue = class 0, red = class 1
    markers = ['o', 's']

    for c in range(2):
        mask = y == c
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=colors[c], marker=markers[c],
                   s=180, zorder=5, edgecolors='white', linewidth=1.2,
                   label=f'Class {c}')

    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(-0.4, 1.4)
    ax.set_xlabel('x₁', fontsize=11)
    ax.set_ylabel('x₂', fontsize=11)
    ax.set_title(title, fontsize=11, pad=8)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.set_aspect('equal')

    if show_line and w[1] != 0:
        x_vals = np.linspace(-0.4, 1.4, 200)
        y_vals = -(w[0] * x_vals + b) / w[1]
        ax.plot(x_vals, y_vals, color='#534AB7', linewidth=2,
                linestyle='--', alpha=0.85, label='Decision boundary')
        ax.legend(fontsize=9, framealpha=0.9)

        # Shade the two regions
        ax.fill_between(x_vals, y_vals, 1.4,
                        alpha=0.06, color='#E24B4A')
        ax.fill_between(x_vals, -0.4, y_vals,
                        alpha=0.06, color='#378ADD')
    elif show_line and w[1] == 0 and w[0] != 0:
        # Vertical boundary
        x_val = -b / w[0]
        ax.axvline(x_val, color='#534AB7', linewidth=2,
                   linestyle='--', alpha=0.85)


def figure_01_learning_steps(gate_name, X, y, perceptron):
    """
    Figure 1: Show the decision boundary at key moments during learning.
    This makes the learning rule tangible — you can see the line moving.
    """
    history = perceptron.history
    n_snaps = min(6, len(history))
    # Pick evenly spaced snapshots
    indices = np.linspace(0, len(history) - 1, n_snaps, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(f'Perceptron learning the {gate_name} gate — decision boundary evolving',
                 fontsize=13, y=1.01)

    for i, (ax, idx) in enumerate(zip(axes.flat, indices)):
        w, b = history[idx]
        step_label = 'Initial (random)' if idx == 0 else f'After update {idx}'
        plot_decision_boundary(ax, w, b, X, y,
                               title=f'Step {i+1}: {step_label}',
                               show_line=True)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'{gate_name.lower()}_learning_steps.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {path}")


def figure_02_final_boundaries():
    """
    Figure 2: Final learned boundaries for AND, OR, and XOR side by side.
    The XOR panel will show that the perceptron simply cannot solve it —
    no matter how long you train, no straight line separates the classes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle('What a perceptron can and cannot learn', fontsize=13, y=1.02)

    results = {}
    for ax, gate in zip(axes, ['AND', 'OR', 'XOR']):
        X, y = make_logic_gate(gate)
        p = Perceptron(n_inputs=2, lr=0.1)
        p.train(X, y, epochs=20)

        preds = p.predict(X)
        acc = (preds == y).mean()
        results[gate] = acc

        status = '✓ Solved' if acc == 1.0 else '✗ Impossible (not linearly separable)'
        color = '#1D9E75' if acc == 1.0 else '#E24B4A'
        ax.set_title(f'{gate} gate\n{status}', fontsize=11, color=color)
        plot_decision_boundary(ax, p.w, p.b, X, y, title='', show_line=True)
        ax.set_title(f'{gate} gate — {status}', fontsize=10, color=color)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'final_boundaries.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {path}")
    return results


def figure_03_why_xor_fails():
    """
    Figure 3: XOR geometrically — the key insight that drove the field forward.

    XOR points:
      (0,0) → 0   (1,1) → 0   : these two need to be on the SAME side
      (0,1) → 1   (1,0) → 1   : these two need to be on the SAME side

    But draw them — the two classes form an X pattern.
    No single straight line can separate them. It's geometrically impossible.

    Solution? Add a hidden layer. Two perceptrons together can solve XOR.
    That realisation eventually led to the MLP, backprop, and modern deep learning.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Why XOR breaks the perceptron — and what it teaches us',
                 fontsize=13, y=1.02)

    X, y = make_logic_gate('XOR')
    colors = ['#378ADD', '#E24B4A']
    labels = ['XOR = 0', 'XOR = 1']

    # Left: the impossible problem
    ax = axes[0]
    for c in range(2):
        mask = y == c
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=colors[c], s=280, zorder=5,
                   edgecolors='white', linewidth=1.5, label=labels[c])
    # Annotate each point
    for xi, yi_val in zip(X, y):
        ax.annotate(f'({int(xi[0])},{int(xi[1])})→{yi_val}',
                    xy=xi, xytext=(xi[0] + 0.07, xi[1] + 0.07),
                    fontsize=9, color='#444')

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title('XOR: try to draw ONE line separating blue from red.\nIt cannot be done.', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_aspect('equal')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')

    # Right: the fix — two hidden neurons create two lines
    ax2 = axes[1]
    for c in range(2):
        mask = y == c
        ax2.scatter(X[mask, 0], X[mask, 1],
                    c=colors[c], s=280, zorder=5,
                    edgecolors='white', linewidth=1.5, label=labels[c])

    x_line = np.linspace(-0.5, 1.5, 200)
    # Two lines that together isolate the two XOR=1 points
    ax2.plot(x_line, x_line + 0.5,  color='#534AB7', lw=2, ls='--', label='Hidden neuron 1 boundary')
    ax2.plot(x_line, x_line - 0.5,  color='#0F6E56', lw=2, ls='--', label='Hidden neuron 2 boundary')

    # Shade the region between them — that's where XOR=1 lives
    ax2.fill_between(x_line,
                     x_line - 0.5,
                     x_line + 0.5,
                     alpha=0.12, color='#E24B4A', label='Region: XOR = 1')

    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_title('Fix: TWO lines (a hidden layer) can isolate XOR=1 region.\nThis is why depth matters.', fontsize=10)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.25)
    ax2.set_aspect('equal')
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'xor_impossibility.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {path}")


def figure_04_weight_trace(gate_name):
    """
    Figure 4: Show how the weights change during training.
    This makes the learning rule concrete — weights are not magic,
    they're numbers that shift a little every time the perceptron is wrong.
    """
    X, y = make_logic_gate(gate_name)
    p = Perceptron(n_inputs=2, lr=0.1)
    p.train(X, y, epochs=20)

    history = p.history
    w0_vals = [h[0][0] for h in history]
    w1_vals = [h[0][1] for h in history]
    b_vals  = [h[1]    for h in history]
    steps   = list(range(len(history)))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f'{gate_name} gate — how weights change during learning', fontsize=12)

    for ax, vals, label, color in zip(
        axes,
        [w0_vals, w1_vals, b_vals],
        ['w₁ (weight for x₁)', 'w₂ (weight for x₂)', 'b (bias)'],
        ['#534AB7', '#1D9E75', '#E8593C']
    ):
        ax.plot(steps, vals, color=color, linewidth=2, marker='o',
                markersize=4, alpha=0.85)
        ax.axhline(0, color='gray', linewidth=0.7, linestyle=':')
        ax.set_xlabel('Update step', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'{gate_name.lower()}_weight_trace.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print()
    print("=" * 60)
    print("  neural-odyssey | Module 00: The Perceptron")
    print("=" * 60)

    # ---- AND gate ----
    print("\n[ AND gate ]")
    X_and, y_and = make_logic_gate('AND')
    p_and = Perceptron(n_inputs=2, lr=0.1)
    p_and.train(X_and, y_and, epochs=20)
    figure_01_learning_steps('AND', X_and, y_and, p_and)
    figure_04_weight_trace('AND')

    # ---- OR gate ----
    print("\n[ OR gate ]")
    X_or, y_or = make_logic_gate('OR')
    p_or = Perceptron(n_inputs=2, lr=0.1)
    p_or.train(X_or, y_or, epochs=20)
    figure_01_learning_steps('OR', X_or, y_or, p_or)

    # ---- XOR gate — the famous failure ----
    print("\n[ XOR gate - this WILL NOT converge ]")
    X_xor, y_xor = make_logic_gate('XOR')
    p_xor = Perceptron(n_inputs=2, lr=0.1)
    p_xor.train(X_xor, y_xor, epochs=20)

    # ---- Summary figure ----
    print("\n[ Building summary figures ]")
    results = figure_02_final_boundaries()
    figure_03_why_xor_fails()

    # ---- Print final summary ----
    print()
    print("=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    for gate, acc in results.items():
        status = "[OK] Solved" if acc == 1.0 else "[FAIL] Not solvable (by design)"
        print(f"  {gate:4s} gate | Accuracy: {acc:.0%} | {status}")

    print()
    print("  KEY TAKEAWAYS")
    print("  -----------------------------------------------------")
    print("  1. A perceptron draws ONE straight line. That's all.")
    print("  2. It can only solve problems where a straight line")
    print("     separates the two classes (linearly separable).")
    print("  3. XOR is not linearly separable. No single perceptron")
    print("     can ever solve it - mathematically impossible.")
    print("  4. The fix: stack TWO perceptrons. A hidden layer.")
    print("     Two lines can isolate the XOR=1 region.")
    print("  5. This insight - that depth helps - eventually led")
    print("     to MLP -> backprop -> deep learning -> transformers.")
    print()
    print("  Output files saved to: module_00_perceptron/output/")
    print("  Next: Module 01 - MLP and the forward pass")
    print("=" * 60)
