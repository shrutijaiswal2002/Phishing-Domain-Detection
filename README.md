# Phishing-Domain-Detection
**Technical Analysis of the Code**  

**1. Machine Learning Model Used**  
- The code is using the **XGBoost (Extreme Gradient Boosting) Classifier**
- **XGBoost** is a powerful gradient boosting algorithm optimized for performance and efficiency. It is widely used for classification and regression tasks, especially for structured/tabular data.

---

 **2. Overview of the Code Workflow**
1. **Importing Necessary Libraries**
   ```python
   import pandas as pd
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score, recall_score, precision_score
   from xgboost import XGBClassifier
   ```
   - **pandas**: Used for data manipulation.
   - **scikit-learn (`sklearn`)**: Used for model training, evaluation, and splitting the dataset.
   - **XGBoost (`XGBClassifier`)**: The primary model for classification.

2. **Defining Features and Target Variable**
   ```python
   X = df.drop(columns=['phishing'], axis=1)
   y = df['phishing']
   ```
   - `X` contains all columns except `"phishing"`, which is the target variable.
   - `y` contains the labels (whether a website is phishing or not).

3. **Splitting Data into Training and Testing Sets**
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
   ```
   - 70% of the data is used for training, and 30% for testing.
   - `random_state=42` ensures reproducibility.
   - `shuffle=True` ensures the dataset is randomly shuffled before splitting.

4. **Training the XGBoost Model**
   ```python
   xgb_clf = XGBClassifier(use_label_encoder=False)
   xgb_clf.fit(X_train, y_train)
   ```
   - The classifier is initialized with `use_label_encoder=False` (to avoid a warning in newer versions of XGBoost).
   - The model is trained using the `.fit()` method.

5. **Making Predictions on Training Data**
   ```python
   preds = xgb_clf.predict(X_train)
   ```
   - The model predicts labels on the **training data**.

6. **Evaluating Training Performance**
   ```python
   print(accuracy_score(y_train, preds))
   print(recall_score(y_train, preds))
   print(precision_score(y_train, preds))
   ```
   - `accuracy_score()`: Measures overall correctness.
   - `recall_score()`: Measures the ability to detect phishing sites (important for security applications).
   - `precision_score()`: Measures the percentage of correctly identified phishing sites among all predicted phishing sites.

7. **Making Predictions on Test Data**
   ```python
   test_preds = xgb_clf.predict(X_test)
   ```
   - The model predicts labels for the **test data**.

8. **Evaluating Test Performance**
   ```python
   print(accuracy_score(y_test, test_preds))
   print(recall_score(y_test, test_preds))
   print(precision_score(y_test, test_preds))
   ```
   - The same metrics are computed for the test set to evaluate generalization performance.

9. **Generating the Confusion Matrix**
   ```python
   from sklearn.metrics import confusion_matrix
   confusion_matrix(y_test, test_preds)
   ```
   - The **confusion matrix** provides detailed insights into how many phishing vs. non-phishing instances were correctly or incorrectly classified.

---

### **Additional Technical Details**
- **Why XGBoost?**  
  - Optimized for speed and performance.
  - Handles missing values automatically.
  - Built-in feature importance ranking.
  - Regularization to prevent overfitting.

- **Potential Improvements**
  - **Hyperparameter tuning:** Use GridSearchCV or RandomizedSearchCV to optimize model performance.
  - **Feature engineering:** Add new features to improve phishing detection.
  - **Cross-validation:** Use `cross_val_score` to ensure stability.

---

Would you like a deeper analysis on feature importance or hyperparameter tuning? 🚀
