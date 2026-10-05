This project demonstrates how to classify heart disease data using standard machine learning models, with a specific focus on optimizing a **Random Forest Classifier** through hyperparameter tuning.

## 🧠 The Concept: How Random Forest Works
A single Decision Tree often memorizes the training data (overfitting). Random Forest fixes this by building a massive "forest" of many trees and having them vote on the final prediction. 

To ensure the trees don't all make the exact same mistakes, the algorithm uses two tricks:
1. **Bootstrapping (Data Randomness):** Each tree is trained on a random subset of the data (controlled by `max_samples`).
2. **Feature Randomness:** At every node split, the tree is only allowed to look at a random subset of the features (controlled by `max_features`). 

## 🛠️ What is Used in this Project
* **Dataset:** `heart.csv` (14 features predicting a binary target: 1 = Disease, 0 = No Disease).
* **Baseline Models (`scikit-learn`):** Logistic Regression, Random Forest, Gradient Boosting, and Support Vector Classifier (SVC).
* **Validation:** Train/Test split (80/20) and K-Fold Cross-Validation.
* **Tuning Methods:**
  * **`GridSearchCV`:** Tests every single combination of a defined hyperparameter grid. Thorough, but computationally expensive.
  * **`RandomizedSearchCV`:** Tests a random sample of combinations. Much faster and often finds near-optimal parameters with a fraction of the compute power.

## 🔑 What is Important: Key Hyperparameters
When tuning a Random Forest, these are the most critical parameters that control the trade-off between bias and variance:

* **`n_estimators` (e.g., 20, 100, 120):** The total number of trees in the forest. More trees generally increase stability but take longer to train.
* **`max_samples` (e.g., 0.5, 0.75):** The percentage of the dataset given to each tree. Lowering this increases diversity among the trees, preventing overfitting. (In this code, manually setting this to 0.75 boosted baseline accuracy to 90%).
* **`max_features` (e.g., 0.2, 0.6):** The percentage of features evaluated at each split. Lower values force the trees to be more diverse. 
* **`max_depth` / `min_samples_leaf`:** Controls how large each individual tree can grow. Restricting depth prevents individual trees from memorizing noise.

## 🚀 Key Takeaway
While Logistic Regression performed very well out-of-the-box on this specific dataset, this code illustrates how leveraging Cross-Validation and Search CV techniques can extract the maximum possible performance from a complex ensemble model like Random Forest while actively preventing it from overfitting.