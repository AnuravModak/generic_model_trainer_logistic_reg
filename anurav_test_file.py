from Model.TrainModel import TrainModel, LogisticRegressionModel
from Utilities.Commons import MainConfiguration
from Global.Variables import config

def main():
    # Model parameters
    trainset_path = config["commons_train_set_path"] +"Airline_customer_satisfaction.csv" # Replace with your actual training data file
    testset_path = config["commons_test_set_path"] +"Airline_customer_satisfaction.csv" # Replace with your actual test data file
    print("Trainset",trainset_path)
    print("TestSet",testset_path)
    result_column = "satisfaction"     # Replace with the target column in your dataset
    model_name = "logistic_regression_model"  # Name for the saved model
    columns_to_encode = ["Customer Type", "Type of Travel","Class","satisfaction" ]  # Categorical columns
    model_flag = 1  # 1 for Logistic Regression, 2 for Decision Tree

    # Initialize training process
    trainer = TrainModel(model_class=LogisticRegressionModel)
    trainer.train_model(trainset_path, result_column, testset_path, model_name, columns_to_encode)

if __name__ == "__main__":
    main()
