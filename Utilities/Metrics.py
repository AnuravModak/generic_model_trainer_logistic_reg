from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import roc_curve, auc
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import log_loss

import  numpy as np


import pandas as pd

class ModelScore:
    def __init__(self,model):
        self.model=model

    def sigmoid_function(self, X):
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        # Calculate z for each data point
        model_coefficients = self.model.coef_[0]
        intercept = self.model.intercept_[0]

        z = np.dot(X, model_coefficients) + intercept
        # Apply sigmoid to each z to get probabilities
        probabilities = sigmoid(z)
        return probabilities

    def my_log_loss(self, X_test, y_test):
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        logloss = log_loss(y_test, y_pred_proba)
        return logloss

    def auc_score(self, X_test,y_test):
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)
        return auc_score

    def variance_influence_factor(self,X_train):
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_train.columns
        vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
        return vif_data

    # def get_SHAP_values(self, X_train,X_test):
    #     explainer = shap.LinearExplainer(self.model, X_train)
    #     shap_values = explainer.shap_values(X_test)
    #
    #     # Visualize the feature importance
    #     shap.summary_plot(shap_values, X_test, plot_type="bar")
    #     return shap_values