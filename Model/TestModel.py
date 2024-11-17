from Global.Variables import config
from Utilities.Commons import Preprocessing


import json
import pandas as pd

from joblib import dump
from joblib import load

class TestModel:
    def __init__(self, model_class):
        """
        Initialize the testing process with a specific model class.

        Parameters:
            model_class: A class implementing the `BaseModel` interface.
        """
        self.model = model_class()  # Instantiate the provided model class
        self.Preprocessing=Preprocessing()


    def get_mapping_real_value(self,df_train_set, df_encoded, column, encoded_value):
        if self.Preprocessing.check_if_string(df_train_set[column]):
            print(f"\nMapping for column '{column}':")
            unique_values = df_train_set[column].unique().tolist()
            unique_values_encoded = df_encoded[column].unique().tolist()
            if isinstance(encoded_value, (str, object)):
                encoded_value = self.Preprocessing.verify_and_parse(encoded_value)
            if encoded_value in unique_values_encoded:
                index = unique_values_encoded.index(encoded_value)
                return unique_values[index]
            else:
                return encoded_value
        else:
            return encoded_value


    def test_model(self,model_name, single_data_point1, result_column):
        loaded_model = None
        base_path = config["commons_models_file_path"] + model_name + os.path.sep

        try:
            loaded_model = load(base_path + model_name + '_log_reg.joblib')
        except FileNotFoundError:
            response = "Model file '" + model_name + ".joblib' not found. Exiting test process."
            print(response)
            return response
        if loaded_model:
            single_data_point = json.loads(json.dumps(single_data_point1))

            # Convert to DataFrame with a single row
            single_df = pd.DataFrame([single_data_point])

            single_df = self.Preprocessing.encode_columns(single_df)

            columns_to_encode = single_df.columns.tolist()
            if result_column in columns_to_encode:
                columns_to_encode.remove(result_column)
            # Select features
            X_single = single_df[columns_to_encode]

            prediction_single = loaded_model.predict(X_single)

            # Display prediction
            print("Prediction for the single data point using the loaded model:")

            decision = self.Preprocessing.get_mapping_real_value(load(base_path + model_name + "_original_cleaned_df.joblib"),
                                              load(base_path + model_name + "_log_reg_encoded_df.joblib"),
                                              result_column, prediction_single[0])
            print(decision)
            return decision