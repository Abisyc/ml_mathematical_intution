# 🌳 Decision Trees: Mathematical Intuition & Splitting Logic

A Decision Tree is a non-parametric supervised learning algorithm that makes predictions by continuously splitting data into smaller, purer piles based on a series of True/False questions.

---

## 1. The Core Intuition

Unlike Linear Regression which draws a single line of best fit, a Decision Tree draws orthogonal boundaries (straight vertical or horizontal lines) to carve the dataset into distinct regions. 

The goal? Find the exact thresholds that separate different classes so perfectly that every resulting pile contains only one type of data (a "Pure Node").

---

## 2. Measuring Messiness: Gini Impurity

To know if a split is good, we must measure how "mixed" or "impure" a pile of data is. The most common metric for classification is **Gini Impurity**.

* **Gini Impurity Formula:**
  $$Gini = 1 - \sum_{i=1}^{C} (p_i)^2$$
  * $C$: Total number of classes (e.g., 2 for binary classification)
  * $p_i$: The probability (or fraction) of items belonging to class $i$ in that pile.

* **Interpretation:**
  * **$Gini = 0.0$**: A perfectly pure node (e.g., 100% Setosa). The algorithm stops splitting here.
  * **$Gini = 0.5$**: The worst possible score for binary classification (a perfectly mixed 50/50 pile).

---

## 3. Optimization: Information Gain

To choose the best question (e.g., "Is Petal Length <= 2.75?"), the algorithm calculates how much "purer" the data gets after the split. This is called **Information Gain (IG)**.

* **Information Gain Formula:**
  $$IG = Gini_{parent} - \left( W_{left} \cdot Gini_{left} + W_{right} \cdot Gini_{right} \right)$$
  * $Gini_{parent}$: Impurity of the data before the split.
  * $W_{left}, W_{right}$: The fraction of total samples that went to the left and right child nodes.
  * $Gini_{left}, Gini_{right}$: Impurity of the new child nodes.

The algorithm tests every possible threshold and crowns the one that yields the **maximum Information Gain** as the winning split.

---

## 4. The Algorithm (How it actually trains)

1. **Find Candidates:** Sort the data for a feature and find the midpoints between every adjacent value. These are your candidate thresholds.
2. **Test Every Split:** For every candidate, split the data into Left and Right.
3. **Calculate IG:** Measure the Gini Impurity of the Left and Right piles, then calculate the Information Gain.
4. **Pick the Winner:** The threshold with the highest Information Gain becomes the official split (the "node").
5. **Repeat Recursively:** Do this for the new Left and Right piles until every pile hits $Gini = 0$ (a Leaf Node) or a max depth is reached.

---

## 5. Summary Formula Cheatsheet

| Metric | Formula | What it means |
| :--- | :--- | :--- |
| **Probability of Class $i$** | $p_i = \frac{\text{count}(class\_i)}{\text{total samples}}$ | Ratio of a specific class in the node |
| **Gini Impurity** | $Gini = 1 - \sum (p_i)^2$ | How mixed the node is (Lower is better) |
| **Node Weight** | $W = \frac{N_{child}}{N_{parent}}$ | Proportion of data that went to the child |
| **Information Gain** | $IG = Gini_{parent} - \sum (W_{child} \cdot Gini_{child})$ | How much purity we gained (Higher is better) |