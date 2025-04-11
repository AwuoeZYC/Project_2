
# XGBoost Classifier

XGBoost (eXtreme Gradient Boosting) is a scalable and efficient implementation of gradient-boosted decision trees, designed for fast learning and high performance on tabular data. Unlike linear models, XGBoost builds an ensemble of decision trees sequentially, where each new tree is trained to correct the errors of the previous ones.

Mathematically, the model predicts:

$$
\hat{y}(\mathbf{x}) = \sum_{m=1}^{M} f_m(\mathbf{x}), \quad f_m \in \mathcal{F}
$$

where each \( f_m \) is a regression tree and \( \mathcal{F} \) is the space of possible trees. The model minimizes a regularized objective:

$$
\mathcal{L} = \sum_{i=1}^{n} \ell(y_i, \hat{y}_i) + \sum_{m=1}^{M} \Omega(f_m)
$$

Here, \( \ell \) is a loss function (e.g., logistic loss), and \( \Omega(f) \) penalizes tree complexity to reduce overfitting.

---

### How We Used XGBoost

In this project, we selected XGBoost as the final model because of its ability to handle non-linear patterns and its built-in support for class imbalance through the `scale_pos_weight` parameter.

We used all available features, including both original variables and those derived through feature engineering. Categorical variables were one-hot encoded using `OneHotEncoder`, and numerical features were passed through unchanged.

To further address class imbalance, we calculated `scale_pos_weight` as the ratio of negative to positive cases in the training data, which was approximately 169. During tuning, we experimented with SMOTE and performed hyperparameter search using `RandomizedSearchCV` to optimize ROC-AUC. After identifying the best parameters, we retrained the final model without SMOTE for better generalization.

The final XGBoost model achieved stronger performance than the baseline logistic regression model, especially in recall and AUC, confirming its suitability for this imbalanced classification task.
