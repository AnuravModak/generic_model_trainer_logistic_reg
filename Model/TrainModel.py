import os
import json
import matplotlib
import pandas as pd

from joblib import dump
from joblib import load

from Global.Variables import config
from abc import ABC , abstractmethod
from Utilities.Commons import JsonOperations, Preprocessing, MainConfiguration

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from Utilities.Metrics import ModelScore

# import shap
matplotlib.use('Agg')


class BaseModel(ABC):
    def __init__(self):
        self.model=None
        self.accuracy=0

    @abstractmethod
    def train(self, X_train,y_train):
        pass

    # @abstractmethod
    # def test(self,X_test,y_test):
    #     pass

    @abstractmethod
    def evaluate(self,X_test, y_test):
        pass



class LogisticRegressionModel (BaseModel):

    def __init__(self,name="LogisticRegressionModel"):
        super().__init__()
        self.model=LogisticRegression(max_iter=1000)
        self.model_name=name
        self.pipeline = make_pipeline(StandardScaler(),self.model )

    def train(self, X_train,y_train):
        self.pipeline.fit(X_train,y_train)

    def evaluate(self,X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {self.accuracy}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        return self.accuracy


class DecisionTreeModel(BaseModel):
    def __init__(self,max_depth,random_state, name="DecisionTreeModel"):
        super().__init__()
        self.model = self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        self.model_name=name
        self.pipeline = make_pipeline(StandardScaler(), DecisionTreeClassifier())

    def train(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {self.accuracy}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        return self.accuracy

class TrainModel:
    def __init__(self, model_class):
        """
        Initialize the training process with a specific model class.

        Parameters:
            model_class: A class implementing the `BaseModel` interface.
        """
        self.model = model_class()  # Instantiate the provided model class

    def train_model(self, trainset_path, result_column, testset_path, model_name, columns_to_encode):
        """
        Train a model (Logistic Regression or Decision Tree) based on the provided configurations and data.

        Args:
            trainset_path (str): Path to the training dataset.
            result_column (str): The target column for prediction.
            testset_path (str): Path to the test dataset.
            model_name (str): Name of the model to be saved.
            columns_to_encode (list): Columns to encode for categorical data.
        """
        main_config = MainConfiguration()
        config = main_config.initialize_environment("../../commons/config.json")

        # Ensure model directory exists
        main_config.preprocessing.create_directory(
            os.path.join(config["commons_models_file_path"], model_name)
        )

        # Load and preprocess data
        df3 = main_config.preprocessing.prerequisites(trainset_path, model_name, columns_to_encode)

        # Identify numerical and categorical features separately
        numerical_features = df3.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = df3.select_dtypes(include=["object", "category"]).columns.tolist()

        # Remove the target column from both feature sets
        if result_column in numerical_features:
            numerical_features.remove(result_column)
        if result_column in categorical_features:
            categorical_features.remove(result_column)

        # Preprocessing: Combine numerical and categorical features
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features),
            ],
            remainder='passthrough'
        )

        # Update X to include all columns except the result_column
        X = df3.drop(columns=[result_column])
        y = df3[result_column]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

        # Model training and saving
        base_path = os.path.join(config["commons_models_file_path"], model_name)

        if self.model.model_name=="LogisticRegressionModel":
            self._train_logistic_regression(X_train, X_test, y_train, y_test, base_path, model_name, result_column, testset_path)
        elif self.model.model_name=="DecisionTreeModel":
            self._train_decision_tree(X_train, X_test, y_train, y_test, base_path, model_name, result_column, testset_path)



    def _train_logistic_regression(
        self, X_train, X_test, y_train, y_test, base_path, model_name, result_column, testset_path
    ):
        log_reg_model = LogisticRegression(max_iter=1000)
        log_reg_model.fit(X_train, y_train)

        # Save the model
        model_path = os.path.join(base_path, f"{model_name}_log_reg.joblib")
        dump(log_reg_model, model_path)

        # Evaluate and print metrics
        print(f"Logistic Regression model trained successfully and saved at {model_path}.")
        print(f"Accuracy: {log_reg_model.score(X_test, y_test)}")
        print("Classification Report:")
        print(classification_report(y_test, log_reg_model.predict(X_test)))

        # Calculate and print loss and other scores
        # Calculate and print loss and other scores
        model_score = ModelScore(log_reg_model)

        # Calculate AUC score
        auc_score = model_score.auc_score(X_test, y_test)
        print(f"AUC Score: {auc_score}")

        # Calculate Log Loss
        log_loss = model_score.my_log_loss(X_test, y_test)
        print(f"Log Loss: {log_loss}")

        # Calculate Variance Inflation Factor (VIF)
        vif_score = model_score.variance_influence_factor(X_train)
        print(f"Variance Inflation Factor (VIF): {vif_score}")

        # Calculate Sigmoid Function Output
        sigmoid_score = model_score.sigmoid_function(X_train)
        print(f"Sigmoid Function Output: {sigmoid_score}")

    def _train_decision_tree(
        self, X_train, X_test, y_train, y_test, base_path, model_name, result_column, testset_path
    ):
        dt_model = DecisionTreeClassifier()
        dt_model.fit(X_train, y_train)

        # Save the model
        model_path = os.path.join(base_path, f"{model_name}_decision_tree.joblib")
        dump(dt_model, model_path)

        # Evaluate and print metrics
        print(f"Decision Tree model trained successfully and saved at {model_path}.")
        print(f"Accuracy: {dt_model.score(X_test, y_test)}")
        print("Classification Report:")
        print(classification_report(y_test, dt_model.predict(X_test)))

