# 🟣 Logistic Regression: Mathematical Intuition & Gradient Descent

This notebook demonstrates the inner workings of Logistic Regression for binary classification. We generate a custom classification dataset, establish a baseline using Scikit-Learn, and then implement the Gradient Descent algorithm entirely from scratch using pure NumPy to prove the math works identically.

---

## 1. The Core Objective
Logistic regression doesn't directly predict classes; it predicts the **probability** that a given point belongs to a specific class. 

To separate two classes on a 2D plane, we need to find the **Decision Boundary**—the best-fit line separating the data points. 

---

## 2. Important Formulas & Mathematical Intuition

### A. The Activation Function (Sigmoid)
To convert a linear equation ($z = W^T X + b$) into a probability between 0 and 1, we pass it through the **Sigmoid** function:
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

### B. The Decision Boundary Line
Our model finds weights ($W_1, W_2$) and a bias/intercept ($W_0$). The decision boundary occurs where the probability is exactly $0.5$, which corresponds to $z = 0$:
$$
W_1 x_1 + W_2 x_2 + W_0 = 0
$$
To plot this line on a standard 2D graph ($y = mx + b$), we isolate $x_2$ (our y-axis):
$$
x_2 = -\frac{W_1}{W_2} x_1 - \frac{W_0}{W_2}
$$
* **Slope ($m$):** $-\frac{W_1}{W_2}$
* **Intercept ($b$):** $-\frac{W_0}{W_2}$

### C. Binary Cross-Entropy (Log Loss)
To evaluate how wrong the predictions are, we minimize the Log Loss function (since Mean Squared Error creates a non-convex space for classification):
$$
J(W) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
$$

### D. Weight Update Rule (Gradient Descent)
By finding the derivative of the loss function, we update the weights iteratively. In this notebook, we use the vector form of the update rule:
$$
W = W + \alpha \frac{X^T (y - \hat{y})}{n}
$$
*(Where $\alpha$ is the learning rate, and $(y - \hat{y})$ is the error).*

---

## 3. Implementation Steps in the Notebook

### Step 1: Dataset Creation
We use `sklearn.datasets.make_classification` to generate 100 sample points with 2 features (so they can be easily plotted on a 2D graph).

### Step 2: Scikit-Learn Baseline
We fit `sklearn.linear_model.LogisticRegression()` to our data. This gives us the target coefficients ($W_1, W_2$) and intercept ($W_0$). We use the Decision Boundary formulas above to plot the best classifying line.

### Step 3: From-Scratch Gradient Descent
We build a custom `gd(x,y)` function that mimics `sklearn`:
1. **Bias Trick:** We insert a column of `1`s into our $X$ matrix so we can calculate the intercept ($W_0$) simultaneously with the weights.
2. **Initialization:** We initialize all weights to `1`.
3. **Training Loop:** We run 5,000 iterations (epochs) with a learning rate (`lr`) of 0.5.
4. **Prediction:** Inside the loop, we calculate $\hat{y}$ using our `sigmoid` function.
5. **Update:** We update the weights using the gradient descent derivative.

### Step 4: Visualizing the Result
Finally, we calculate the slope and intercept for our custom weights and plot them against the Scikit-Learn line. The **red line** (sklearn) and **black line** (our math) overlap perfectly, proving our custom gradient descent converged on the absolute minimum!

---

## 🚀 Future Outlook
While it takes a solid understanding of calculus and linear algebra to build this with pure NumPy, modern deep learning frameworks abstract this away. As noted in the notebook, this exact optimization process can be replicated in just ~10 lines of code using **PyTorch**.