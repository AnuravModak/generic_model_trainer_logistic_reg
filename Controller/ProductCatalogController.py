import os
import json
import numpy as np
from flask import Flask, request, jsonify, Blueprint, send_file
from flask_cors import CORS
from Model.TrainModel import TrainModel
from Model.TestModel import TestModel
from Global.Variables import config

from Utilities.Commons import JsonOperations, CorrelationMatrix


class ProductAPI:
    def __init__(self, url_prefix="/api", host="0.0.0.0", port=7654, debug=True):
        """
        Initialize the Product API.

        Parameters:
            url_prefix (str): The URL prefix for the API endpoints.
            host (str): The host address for the Flask app.
            port (int): The port for the Flask app.
            debug (bool): Enable or disable debug mode.
        """
        self.app = Flask(__name__)
        self.url_prefix = url_prefix
        self.host = host
        self.port = port
        self.debug = debug

        # Enable CORS
        CORS(self.app)

        # Register blueprint
        self.product_bp = Blueprint("product_bp", __name__)
        self.register_routes()
        self.app.register_blueprint(self.product_bp, url_prefix=self.url_prefix)

        # Add health check route
        self.app.add_url_rule("/health", "health_check", self.health_check, methods=["GET"])

        # Add Other Local functionalities into this class

        self.json_operations = JsonOperations()

    def register_routes(self):
        """Register API routes for the blueprint."""
        self.product_bp.add_url_rule(
            "/train-model", "train_model_api", self.train_model_api, methods=["POST"]
        )
        self.product_bp.add_url_rule(
            "/test-model", "test_model_api", self.test_model_api, methods=["POST"]
        )
        self.product_bp.add_url_rule(
            "/get-all-products", "get_all_products", self.get_all_products, methods=["GET"]
        )
        self.product_bp.add_url_rule(
            "/re-train-all", "re_train_all_models", self.re_train_all_models, methods=["GET"]
        )
        self.product_bp.add_url_rule(
            "/get-product", "get_product_by_id", self.get_product_by_id, methods=["GET"]
        )
        self.product_bp.add_url_rule(
            "/get-correlation-matrix-image-df",
            "get_correlation_plot_image_df",
            self.get_correlation_plot_image_df,
            methods=["POST"],
        )
        self.product_bp.add_url_rule(
            "/get-correlation-matrix-image",
            "get_correlation_plot_image",
            self.get_correlation_plot_image,
            methods=["GET"],
        )
        self.product_bp.add_url_rule(
            "/get-correlation-matrix-info",
            "get_correlation_plot_text",
            self.get_correlation_plot_text,
            methods=["POST"],
        )

    def train_model_api(self):
        data = request.json
        try:
            columns_to_encode = data.get("encodedColumns", "")

            #add one functionality where we need to create an instance of TrainModel based on model we are passing

            results = train_model(
                data["trainFilePath"],
                data["decisionColumn"],
                data["testFilePath"],
                data["modelName"],
                columns_to_encode,
                int(data["modelFlag"]),
            )
            return jsonify(results), 200
        except Exception as e:
            return jsonify({"error": str(e.__cause__)}), 500

    def test_model_api(self):
        data = request.json
        try:
            single_data_point = data["single_data_point"]
            model_name = data["model_name"]
            decision_column = data["decision_column"]

            if model_name and single_data_point and decision_column:
                #add one more condition to check the model name and based on it create an instance of TestModel class and proceed
                results = test_model(model_name, single_data_point, decision_column)
                return jsonify({decision_column: int(results) if isinstance(results, (np.int64, int)) else results}), 200
            return jsonify({"error": "Invalid request"}), 400
        except Exception as e:
            return jsonify({"error": str(e.__cause__)}), 500

    def get_all_products(self):
        products = self.json_operations.get_product_list_json()
        return jsonify(products), 200

    def re_train_all_models(self):
        products = self.json_operations.get_product_list_json()
        # add one more condition to check the model name and based on it create an instance of TestModel class and proceed
        for product in products:
            train_model(
                product["trainModelPath"],
                product["decisionColumn"],
                product["testModelPath"],
                product["name"],
                product["encodedColumns"],
                product["modelFlag"],
            )
        return jsonify(self.json_operations.get_product_list_json()), 200

    def get_product_by_id(self):
        product_id = request.args.get("id")
        if not product_id:
            return jsonify({"error": "ID query parameter is required"}), 400
        return jsonify(self.json_operations.get_product_json(int(product_id))), 200

    def get_correlation_plot_image_df(self):
        data = request.json
        try:
            model_name = data["model_name"]
            trainset_file_name = data["train_file_name"]
            file_list = os.listdir(config["commons_train_set_path"])
            file_path = next(
                (os.path.join(config["commons_train_set_path"], val) for val in file_list if trainset_file_name in val),
                "",
            )

            if not file_path:
                return jsonify({"error": "File not found"}), 500
            corr_matrix=CorrelationMatrix()
            img = corr_matrix.get_correlation_matrix_image(file_path, model_name)
            return send_file(img, mimetype="image/png")
        except Exception as e:
            return jsonify({"error": str(e.__cause__)}), 500

    def get_correlation_plot_image(self):
        product_id = request.args.get("id")
        if not product_id:
            return jsonify({"error": "ID query parameter is required"}), 400
        product = self.json_operations.get_product_json(int(product_id))
        corr_matrix=CorrelationMatrix()
        img = corr_matrix.get_correlation_matrix_image(product["trainModelPath"], product["name"])
        return send_file(img, mimetype="image/png")

    def get_correlation_plot_text(self):
        data = request.json
        trainset_name = data["train_set_name"]
        model_name = data["model_name"]
        columns_to_encode = data["columns_to_encode"]
        result_column = data["result_column"]
        number_of_train_data = data["number_of_train_data"]

        if not all([trainset_name, model_name, number_of_train_data]):
            return jsonify({"error": "Improper request"}), 400
        corr = CorrelationMatrix()
        corr_mat = corr.get_correlation_matrix(
            trainset_name, model_name, columns_to_encode, result_column, number_of_train_data
        )
        return jsonify({"corr_mat": str(corr_mat)}), 200

    def health_check(self):
        return "Server is up and running!", 200

    def run(self):
        self.app.run(debug=self.debug, port=self.port, host=self.host)


# Initialize and run the app
# if __name__ == "__main__":
#     api = ProductAPI()
#     api.run()
