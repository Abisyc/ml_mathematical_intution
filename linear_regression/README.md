# 📈 Linear Regression: Mathematical Intuition & Derivation

Linear Regression models the relationship between an independent variable ($X$) and a dependent variable ($y$) by fitting a straight line through data points.

---

## 1. The Core Intuition

Imagine plotting a scatter of points on a 2D graph. Linear Regression finds the line that stays as close as possible to all points simultaneously. 

* **Prediction Line Equation:**
  $$\hat{y} = w \cdot x + b$$
  * $\hat{y}$: The predicted value
  * $x$: The input feature
  * $w$ (weight/slope): How much $\hat{y}$ changes when $x$ changes by 1 unit
  * $b$ (bias/intercept): The value of $\hat{y}$ when $x = 0$

---

## 2. Ordinary Least Squares (OLS) Closed-Form Solution

For simple linear regression (like your CGPA vs. Package notebook), we calculate the optimal slope ($m$) and intercept ($b$) using covariance and variance:

* **Slope ($m$):**
  $$m = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}$$

* **Intercept ($b$):**
  $$b = \bar{y} - m \cdot \bar{x}$$

---

## 3. Measuring Error: Mean Squared Error (MSE)

To evaluate performance, we measure how far off predictions are from actual values using Mean Squared Error:

$$J(w, b) = \frac{1}{2n} \sum_{i=1}^{n} \left( \hat{y}^{(i)} - y^{(i)} \right)^2$$

---

## 4. Summary Formula Cheatsheet

| Concept | Formula |
| :--- | :--- |
| **Hypothesis** | $\hat{y} = w x + b$ |
| **Slope ($m$)** | $\frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}$ |
| **Intercept ($b$)** | $\bar{y} - m \bar{x}$ |
| **Cost Function (MSE)** | $\frac{1}{2n} \sum (\hat{y} - y)^2$ |
