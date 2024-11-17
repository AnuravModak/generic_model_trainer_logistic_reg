# Essential Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Machine Learning Imports
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Utilities
from joblib import dump, load
import os
import json
import re
import io
import warnings
from Global.Variables import config
warnings.filterwarnings('ignore')

class Preprocessing:
    def __init__(self):
        """
        Initialize the preprocessing class with common models file path.
        """
        self.commons_models_file_path = config["commons_models_file_path"]

    def load_data(self, file_path):
        """
        Load the dataset from the given file path.
        """
        return pd.read_csv(file_path)

    def create_directory(self,directory_path):
        try:
            os.makedirs(directory_path, exist_ok=True)
            print(f"Directory '{directory_path}' created successfully.")
        except OSError as error:
            print(f"Error creating directory '{directory_path}': {error}")

    def data_cleaning(self, df):
        """
        Clean the dataset by replacing "unknown" with NA, dropping missing values, and resetting the index.
        """
        df.isnull().sum()
        df_cleaned = df.replace("unknown", pd.NA)
        df_cleaned.dropna(inplace=True)
        df_cleaned.reset_index(drop=True, inplace=True)
        return df_cleaned

    @staticmethod
    def get_unique(df_column):
        """
        Get unique values from a column.
        """
        return df_column.unique().tolist()

    @staticmethod
    def check_unique(df_column):
        """
        Check if a column has unique values.
        """
        dup = df_column.duplicated(keep=False)
        df_column['is_duplicate'] = dup
        has_duplicates = df_column['is_duplicate'].any()
        return not has_duplicates

    @staticmethod
    def encode_column(column):
        """
        Encode a single column into numerical values.
        """
        try:
            unique_values = column.unique().tolist()
            return column.map({val: idx for idx, val in enumerate(unique_values)})
        except Exception as e:
            print(f"An error occurred during encoding: {e}")
            encoder = OneHotEncoder()
            return pd.DataFrame(encoder.fit_transform(column.values.reshape(-1, 1)).toarray())

    @staticmethod
    def check_if_string(column):
        """
        Check if a column contains string values.
        """
        return column.dtype == 'object'

    @staticmethod
    def is_numeric(value):
        """
        Check if a value is numeric.
        """
        pattern = re.compile(r'^-?\d+(\.\d+)?$')
        return bool(pattern.match(str(value)))

    @staticmethod
    def check_column_if_numeric(column):
        """
        Check if a column is numeric or contains numeric values stored as strings.
        """
        if pd.api.types.is_numeric_dtype(column):
            return True
        return all(Preprocessing.is_numeric(value) for value in column)

    def encode_columns(self, df):
        """
        Encode all string columns in the dataset into numeric values.
        """
        df_encoded = df.copy()
        for col in df.columns:
            if self.check_if_string(df[col]) and not self.check_column_if_numeric(df[col]):
                df_encoded[col] = self.encode_column(df[col])
        return df_encoded

    def verify_and_parse(self,value):
        try:
            # Try to convert to an integer first
            int_value = int(value)
            print(f"'{value}' is an integer.")
            return int_value
        except ValueError:
            try:
                # If integer conversion fails, try to convert to a float
                float_value = float(value)
                print(f"'{value}' is a float.")
                return float_value
            except ValueError:
                # If both conversions fail, return the original string
                print(f"'{value}' is neither an integer nor a float.")
                return value

    def get_mapping_real_value(self,df_train_set, df_encoded, column, encoded_value):
        if self.check_if_string(df_train_set[column]):
            print(f"\nMapping for column '{column}':")
            unique_values = df_train_set[column].unique().tolist()
            unique_values_encoded = df_encoded[column].unique().tolist()
            if isinstance(encoded_value, (str, object)):
                encoded_value = self.verify_and_parse(encoded_value)
            if encoded_value in unique_values_encoded:
                index = unique_values_encoded.index(encoded_value)
                return unique_values[index]
            else:
                return encoded_value
        else:
            return encoded_value

    def make_training_directory(self, directory_name):
        """
        Create a training directory if it does not already exist.
        """
        directory_path = os.path.join(os.getcwd(), self.commons_models_file_path + directory_name)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Directory '{directory_name}' created successfully.")
        else:
            print(f"Directory '{directory_name}' already exists.")

    def prerequisites(self, trainset_path, model_name,columns_to_encode=None):
        """
        Load, clean, and encode the dataset, then save the cleaned dataset.
        """
        df = self.load_data(trainset_path)
        self.make_training_directory(model_name)
        base_path = self.commons_models_file_path + model_name + os.path.sep

        df_cleaned = self.data_cleaning(df)
        dump(df_cleaned, base_path + model_name + '_original_cleaned_df.joblib')

        if columns_to_encode is None:
            df_encoded = self.encode_columns(df_cleaned)
        else:
            df_encoded = df_cleaned.copy()
            for col in columns_to_encode:
                if col in df_cleaned.columns:
                    df_encoded[col] = self.encode_column(df_cleaned[col])
                else:
                    print(f"Warning: Column '{col}' not found in the dataset. Skipping encoding.")

        print(df_encoded.head())
        return df_encoded

class MainConfiguration:
    def __init__(self):

        self.preprocessing=Preprocessing()


    def load_config(self,config_path):
        """
        Load the configuration JSON file.
        """
        with open(config_path, 'r') as config_file:
            return json.load(config_file)


    def initialize_directories(self,directories):
        """
        Create necessary directories if they don't exist.
        """
        for directory in directories:
            self.preprocessing.create_directory(directory)


    # -------------------------------
    # Data Setup and Initialization
    # -------------------------------

    def initialize_environment(self,config_path):
        """
        Initialize the environment by loading configurations and setting up directories.
        """
        config = self.load_config(config_path)

        directories = [
            config["commons_file_path"],
            config["commons_train_set_path"],
            config["commons_test_set_path"],
            config["commons_models_file_path"]
        ]
        self.initialize_directories(directories)
        return config


    def initialize_data(self):
        """
        Initialize placeholders for training data and new product structure.
        """
        X_train = pd.DataFrame()
        y_train = pd.DataFrame()
        new_product = {
            "id": 0,
            "name": "",
            "accuracy": 0
        }
        return X_train, y_train, new_product

class JsonOperations:

    def __init__(self, file_path="products.json"):
        self.file_path=file_path
        self.products = [] #Initial product list


    def initialise_json(self):
        # Write the initial list of products to a JSON file
        with open(self.file_path, 'w') as file:
            json.dump(self.products, file, indent=4)

    # Function to create a JSON file with initial products if it doesn't exist
    def create_initial_json(self,products):
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as file:
                json.dump(products, file, indent=4)
            # print(f"Created {file_path} with initial products.")
        # else:
        #     print(f"{file_path} already exists.")

    def read_json(self):
        # Verify the update
        self.create_initial_json(self.products)
        with open(self.file_path, 'r') as file:
            updated_products = json.load(file)
            # print(json.dumps(updated_products, indent=4))

        return updated_products

    def product_exists(self,name):
        products = self.get_product_list_json()
        if isinstance(products, tuple):  # Check if it returned an error
            return products
        matching_products = [product for product in products if product['name'].lower() == name.lower()]
        return True if matching_products else False

    def add_product(self,new_product):
        # Function to add a new product to the JSON file
        # Read the existing products from the file

        products = self.read_json()

        # Add the new product to the list
        if len(products) > 0:
            new_product["id"] = products[-1]["id"] + 1
        else:
            new_product["id"] = 1
        if not self.product_exists(new_product["name"]):
            products.append(new_product)

        # Write the updated list back to the file
        with open(self.file_path, 'w') as file:
            json.dump(products, file, indent=4)

        new_product = self.read_json()
        return new_product[-1]

    # Function to remove a product from the JSON file
    def remove_product(self,product_id):
        # Read the existing products from the file
        products = self.read_json()

        # Remove the product with the specified id
        products = [product for product in products if product['id'] != product_id]

        # Write the updated list back to the file
        with open(self.file_path, 'w') as file:
            json.dump(products, file, indent=4)

    def get_attribute_value_arr(self,products, attribute_name):
        product_attr = [product[attribute_name] for product in products]
        return product_attr

    def get_product_json(self,id):
        products = self.read_json()
        if id in self.get_attribute_value_arr(products, "id"):
            return products[id - 1]

    def get_product_json_name(self,name):
        products = self.read_json()
        # Iterate through products and find the one with the matching name
        for product in products:
            if product["name"] == name:
                return product
        # Return None if no product is found with the given name
        return None

    def get_product_list_json(self):
        products = self.read_json()
        return products

    # New product to add
    # new_product = {
    #     "id": 3,
    #     "name": "Tablet",
    #     "accuracy": 89.8
    # }

    # Add the new product to the JSON file
    # add_product(file_path, new_product)
    # remove_product(file_path, 1)

    # print(get_product_list_json())

class CorrelationMatrix:

    def __init__(self, commons_trainset_file_path=config["commons_train_set_path"]):
        self.Preprocessing=Preprocessing()
        self.commons_trainset_file_path = commons_trainset_file_path


    def get_correlation_matrix(self,trainset_name, model_name, columns_to_encode, result_column, number_of_train_data):
        trainset_path = self.commons_trainset_file_path + trainset_name
        self.Preprocessing.make_training_directory(model_name)
        df = self.Preprocessing.prerequisites(trainset_path, model_name)
        df = df.head(int(number_of_train_data))
        if len(columns_to_encode) == 0:
            columns_to_encode = df.columns.tolist()
        if result_column and result_column in columns_to_encode:
            columns_to_encode.remove(result_column)

        corr_mat = self.print_correlation_matrix(df)
        return corr_mat


    def print_correlation_matrix(self,df):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)  # Set a large width to display all columns
        corr_matrix = df.corr()
        print("Correlation Matrix:")
        # Convert correlation matrix to string with formatting
        corr_str = corr_matrix.to_string()
        # Print the formatted correlation matrix
        print(corr_str)
        print("---------------------------------------------------------")
        return corr_matrix


    # def plot_correlation_matrix_plot(df, model_name):
    #     df = prerequisites(trainset_path, model_name)
    #     corr_matrix = df.corr()
    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    #     plt.title(f'Correlation Matrix for {model_name}')
    #     # plt.show()
    #     buf = io.BytesIO()
    #     plt.savefig(buf, format='png')
    #     buf.seek(0)
    #     plt.close()
    #     return buf


    def get_correlation_matrix_image(self,trainset_path, model_name):
        df = self.Preprocessing.prerequisites(trainset_path, model_name)
        corr_matrix = df.corr()
        num_vars = len(corr_matrix.columns)
        fig_size = (num_vars, num_vars * 0.8)
        plt.figure(figsize=fig_size)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
        plt.title(f'Correlation Matrix for {model_name}')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf