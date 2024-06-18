from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


from joblib import dump
from joblib import load
from json_operations import add_product

import pandas as pd
import re
import json
import os
import io
import warnings


warnings.filterwarnings('ignore')

X_train = pd.DataFrame()

y_train = pd.DataFrame()


# Load configuration from JSON file
with open('../commons/config.json', 'r') as config_file:
    config = json.load(config_file)

import os

def create_directory(directory_path):
    try:
        os.makedirs(directory_path, exist_ok=True)
        print(f"Directory '{directory_path}' created successfully.")
    except OSError as error:
        print(f"Error creating directory '{directory_path}': {error}")


commons_file_path = config["commons_file_path"]
commons_models_file_path = config["commons_models_file_path"]
commons_train_set_file_path = config["commons_train_set_path"]
commons_test_set_file_path = config["commons_test_set_path"]

create_directory(commons_file_path)
create_directory(commons_train_set_file_path)
create_directory(commons_test_set_file_path)


new_product = {
    "id": 0,
    "name": "",
    "accuracy": 0
}

def load_data(file_path):
    df1 = pd.read_csv(file_path)
    #print(df1)
    return df1


def data_cleaning(df1):
    df1.isnull().sum()
    df2 = df1.replace("unknown", pd.NA)
    df2.dropna(inplace=True)
    df2.reset_index(drop=True, inplace=True)
    print(df2.columns)
    return df2


def get_unique(df2):
    return df2.unique().tolist()


def check_unique(df):  # pass df['column_name']
    dup = df.duplicated(keep=False)
    df['is_duplicate'] = dup
    has_duplicates = df['is_duplicate'].any()

    if has_duplicates:
        print("There are duplicates in the DataFrame.")
        return False
    else:
        print("All values are unique in the DataFrame.")
        return True


def encode_column(df):
    try:
        arr = get_unique(df)
        output = list()
        for x in df:
            if x in arr:
                output.append(arr.index(x))
        out = pd.DataFrame(output)
        return out
    except Exception as e:
        print(f"An error occurred during encoding: {e}")
        encoder = OneHotEncoder()
        encoded_data = encoder.fit_transform(df)
        return encoded_data
def check_if_string(df):
        # Check the dtype of the column
        column_dtype = df.dtype

        # Determine if the column contains string or numeric values
        if column_dtype == 'object':
            # print(f"Column contains string values.")
            return True
        elif pd.api.types.is_numeric_dtype(df):
            # print(f"Column contains numeric values.")
            return False
        else:
            # print(f"Column contains other types of data.")
            return False


###--------------------------------------
def prerequisites(trainset_path, model_name):
    df1 = load_data(trainset_path)
    make_training_directory(model_name)
    base_path = commons_models_file_path+model_name+os.path.sep

    print(df1)

    df2 = data_cleaning(df1)

    dump(df2, base_path+model_name+'_original_cleaned_df.joblib')

    df3 = encode_columns(df2)

    print(df3.head())
    return df3


def encode_columns(df2):
    columns_to_encode = df2.columns.tolist()
    df3 = df2.copy()
    for col in columns_to_encode:
        if check_if_string(df2[col]) and not check_column_if_numeric(df2[col].values):
            df3[col] = encode_column(df2[col])
    return df3

def run_decision_tree(df3, X_train, X_test, y_train, y_test, result_column, testset_path, model_name):
    global accuracy1
    pipelines = {
        'Decision Tree': make_pipeline(StandardScaler(), DecisionTreeClassifier()),
    }
    for name, pipeline in pipelines.items():
        print(f"Training and evaluating {name}...")
        # Train the model using the pipeline
        pipeline.fit(X_train, y_train)
        # Make predictions on the testing set
        y_pred = pipeline.predict(X_test)
        # Evaluate the classifier's performance
        accuracy1 = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy1)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("---------------------------------------------------------")

    # Train each model
    for name, pipeline in pipelines.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)

    # Extract feature importances or coefficients (if applicable)
    feature_importances = {}
    for name, pipeline in pipelines.items():
        if 'Decision Tree' in name:
            feature_importances[name] = pipeline.named_steps['decisiontreeclassifier'].feature_importances_

    return accuracy1

def run_logistic_regression(df3, X_train, X_test, y_train, y_test, result_column, testset_path, model_name):
    global accuracy
    pipelines = {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    }
    for name, pipeline in pipelines.items():
        print(f"Training and evaluating {name}...")
        # Train the model using the pipeline
        pipeline.fit(X_train, y_train)
        # Make predictions on the testing set
        y_pred = pipeline.predict(X_test)
        # Evaluate the classifier's performance
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("---------------------------------------------------------")

    # Train each model
    for name, pipeline in pipelines.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)

    # Extract feature importances or coefficients
    feature_importances = {}
    for name, pipeline in pipelines.items():
        if 'Logistic Regression' in name:
            feature_importances[name] = pipeline.named_steps['logisticregression'].coef_[0]

    # df_test = prerequisites(testset_path, model_name)
    # columns_to_encode = df_test.columns.tolist()
    # X_test = df_test[columns_to_encode]
    # y_test = df_test[result_column]
    #
    # X_train = df3[columns_to_encode]
    # y_train = df3[result_column]

    return accuracy

def make_training_directory(directory_name):

    # Specify the path where you want to create the directory
    directory_path = os.path.join(os.getcwd(), commons_models_file_path + directory_name)

    # Check if the directory already exists
    if not os.path.exists(directory_path):
        # Create the directory
        os.makedirs(directory_path)
        print(f"Directory '{directory_name}' created successfully.")
    else:
        print(f"Directory '{directory_name}' already exists.")

import pandas as pd

def print_correlation_matrix(df):
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


def plot_correlation_matrix_plot(df, model_name):
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title(f'Correlation Matrix for {model_name}')
    plt.show()

def get_correlation_matrix_image(trainset_path, model_name):
    df = prerequisites(trainset_path, model_name)
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

def train_model(trainset_path, result_column, testset_path, model_name, columns_to_encode, model_flag):
    trainset_path = commons_train_set_file_path + trainset_path
    testset_path = commons_test_set_file_path + testset_path
    make_training_directory(model_name)
    df3 = prerequisites(trainset_path, model_name)
    if len(columns_to_encode) == 0:
        columns_to_encode = df3.columns.tolist()
    if result_column in columns_to_encode:
        columns_to_encode.remove(result_column)
    X = df3[columns_to_encode]
    y = df3[result_column]

    correlation_matrix = print_correlation_matrix(df3)

    # plot_correlation_matrix_plot(df3[columns_to_encode], model_name)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    if model_flag == 1:
         accuracy_score = run_logistic_regression(df3, X_train, X_test, y_train, y_test, result_column, testset_path, model_name)
         # Fit the model to the training data
         log_reg_model = LogisticRegression(max_iter=1000)
         log_reg_model.fit(X_train, y_train)
         base_path = commons_models_file_path+model_name + os.path.sep
         model_path = base_path + model_name + "_log_reg" + ".joblib"
         # Save the trained model
         dump(log_reg_model, model_path)

         dump(df3, base_path + model_name + "_log_reg" + '_encoded_df.joblib')

         print("Model trained and saved successfully at:", model_path)
    elif model_flag == 2:
        accuracy_score = run_decision_tree(df3, X_train, X_test, y_train, y_test, result_column, testset_path, model_name)
        log_reg_model = DecisionTreeClassifier()
        log_reg_model.fit(X_train, y_train)
        base_path = commons_models_file_path + model_name + os.path.sep
        model_path = base_path + model_name + "_decision_tree" + ".joblib"
        # Save the trained model
        dump(log_reg_model, model_path)
        dump(df3, base_path + model_name + "_decision_tree" + '_encoded_df.joblib')

    new_product["name"] = model_name
    new_product["accuracy"] = accuracy_score
    # new_product['correlationMatrix'] = str(correlation_matrix)
    new_product['encodedColumns'] = columns_to_encode
    new_product['trainModelPath'] = trainset_path
    new_product['testModelPath'] = testset_path
    new_product['decisionColumn'] = result_column
    new_product['modelFlag'] = model_flag

    new_prod = add_product(new_product)
    return new_prod

def verify_and_parse(value):
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

def get_mapping_real_value(df_train_set, df_encoded, column, encoded_value):
    if check_if_string(df_train_set[column]):
        print(f"\nMapping for column '{column}':")
        unique_values = df_train_set[column].unique().tolist()
        unique_values_encoded = df_encoded[column].unique().tolist()
        if isinstance(encoded_value, (str, object)):
            encoded_value = verify_and_parse(encoded_value)
        if encoded_value in unique_values_encoded:
            index = unique_values_encoded.index(encoded_value)
            return unique_values[index]
        else:
            return encoded_value
    else:
        return encoded_value



def test_model(model_name, single_data_point1, result_column):
    loaded_model = None
    base_path = commons_models_file_path + model_name+os.path.sep

    try:
        loaded_model = load(base_path+model_name+'.joblib')
    except FileNotFoundError:
        print("Model file '"+model_name+".joblib' not found. Exiting test process.")
        return
    if loaded_model:
        single_data_point = json.loads(json.dumps(single_data_point1))

        # Convert to DataFrame with a single row
        single_df = pd.DataFrame([single_data_point])

        single_df = encode_columns(single_df)

        columns_to_encode = single_df.columns.tolist()
        if result_column in columns_to_encode:
            columns_to_encode.remove(result_column)
        # Select features
        X_single = single_df[columns_to_encode]

        prediction_single = loaded_model.predict(X_single)

        # Display prediction
        print("Prediction for the single data point using the loaded model:")

        decision = get_mapping_real_value(load(f"{model_name}/{model_name}_original_cleaned_df.joblib"), load(f"{model_name}/{model_name}_encoded_df.joblib"), result_column, prediction_single[0])
        print(decision)


def is_numeric(s):
    pattern = re.compile(r'^-?\d+(\.\d+)?$')
    return bool(pattern.match(s))

def check_column_if_numeric(df):
    for column in df:
        if pd.api.types.is_numeric_dtype(column):
            return True
        else:
            # Check if the column contains numeric values stored as strings
            if is_numeric(column):
                return True
            else:
                return False

def convert_into_numeric_values(single_data_point_test_1):
    for key, value in single_data_point_test_1.items():
        if isinstance(value, (int, float)):
            single_data_point_test_1[key] = str(value)
    return single_data_point_test_1


