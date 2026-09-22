# 🛣️ Support Vector Machines (SVM): Math Intuition & Implementation

This folder contains the mathematical breakdown and a pure-NumPy implementation of the Support Vector Machine algorithm. By avoiding high-level libraries like `scikit-learn`, this notebook demonstrates exactly how Gradient Descent optimizes the Maximum Margin Hyperplane.

## 🧠 Mathematical Intuition

Support Vector Machines aim to find a decision boundary that separates classes while maintaining the widest possible distance from the nearest data points (the Support Vectors).

### 1. The Unified Margin Constraint
SVM relies on strict geometric placement using class labels $+1$ and $-1$. 
* **Positive Gutter ($\pi+$):** $\mathbf{w}^T\mathbf{x}_i + b \ge 1$
* **Negative Gutter ($\pi-$):** $\mathbf{w}^T\mathbf{x}_i + b \le -1$

Multiplying the boundary equations by the target label $y_i$ yields a single constraint for correctly classified points outside the margin:

$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1$$

### 2. Maximizing the Margin
The total physical width of the "street" separating the two classes is derived by projecting the distance between the gutters onto the weight vector $\mathbf{w}$:

$$d = \frac{2}{\vert{}\vert{}\mathbf{w}\vert{}\vert{}}$$

To maximize this margin distance, the optimization algorithm must minimize $\frac{1}{2}\vert{}\vert{}\mathbf{w}\vert{}\vert{}^2$.

---

## 🚧 Hard Margin vs. Soft Margin

* **Hard Margin:** Assumes the data is perfectly linearly separable. It strictly enforces that *zero* data points are allowed inside the margin or on the wrong side of the decision boundary. A single outlier can drastically skew the boundary or make the optimization impossible to solve.
* **Soft Margin:** Acknowledges that real-world data has noise and overlaps. It introduces "slack variables" ($\xi$) that allow certain points to violate the margin constraints without breaking the algorithm, trading a little bit of classification error for a much more robust and generalized model.

### The Soft Margin Objective (Hinge Loss)
To implement the soft margin, we introduce a penalty $C$ for points that cross the margin. The final objective function balances widening the street with minimizing these errors:

$$J(\mathbf{w}, b) = \frac{1}{2} \vert{}\vert{}\mathbf{w}\vert{}\vert{}^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))$$

---

## ⚖️ Support Vectors & The $C$ Hyperparameter

**The Role of Support Vectors:**
The algorithm's name comes from the fact that *only* the points sitting exactly on the margin (or violating it) influence the position of the decision boundary. These points are the **Support Vectors**. If you delete all the perfectly classified points far away from the margin, the SVM boundary will not move at all.

**How $C$ Controls Margin Distance ($d$):**
The hyperparameter $C$ dictates how aggressively the model punishes classification errors (points inside the margin). 

| $C$ Value | Model Behavior | Effect on Margin Distance ($d$) |
| :--- | :--- | :--- |
| **High $C$** (e.g., $100$) | Strict. Heavy penalty for errors. The model sacrifices margin width to ensure almost all points are correctly classified. Prone to overfitting. | **Smaller $d$** (Narrow street) |
| **Low $C$** (e.g., $0.01$) | Relaxed. Low penalty for errors. The model prioritizes finding the widest possible margin, ignoring individual noisy points. Better generalization. | **Larger $d$** (Wide street) |

---

## ⚙️ Implementation Details

The `SVM` class is built using pure Python and `numpy`. The training loop uses Stochastic Gradient Descent (SGD) to iteratively evaluate the Hinge Loss and update parameters.

| Component | Description |
| :--- | :--- |
| **`fit(X, y)`** | Evaluates the unified margin condition. Updates weights using partial derivatives only if a point violates the margin. |
| **`predict(X)`** | Returns $+1$ or $-1$ based on the sign of $\mathbf{w}^T\mathbf{x} + b$. |
| **`calculate_margin_distance()`** | Computes the physical width of the maximum margin using $d = \frac{2}{\vert{}\vert{}\mathbf{w}\vert{}\vert{}}$. |