import json

from flask import Flask, request, jsonify, Blueprint, send_file
from flask_cors import CORS
from json_operations import *
from train_model import *

url_prefix = '/api'

app = Flask(__name__)

CORS(app)

# Define the blueprint
product_bp = Blueprint('product_bp', __name__)


@product_bp.route('/train-model', methods=['POST'])
def train_model_api():
    data = request.json
    try:
        # Call the train_model function with the provided data
        columns_to_encode = data["encodedColumns"]
        results = train_model(data["trainFilePath"], data["decisionColumn"], data["testFilePath"], data["modelName"],
                              columns_to_encode, int(data["modelFlag"]))

        # if isinstance(results, tuple):
        #     # If results is a tuple, convert it to a JSON response
        #     response = {result if (i + 1) % 2 != 0 else f"result_{i+1}" : result for i, result in enumerate(results)}
        # else:
        #     # If results is a single value, return it directly
        #     response = {"accuracy": results}

        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e.__cause__)}), 500


@product_bp.route('/get-all-products', methods=['GET'])
def get_all_products():
    products = get_product_list_json()
    return jsonify(products), 200


@product_bp.route('/re-train-all', methods=['GET'])
def re_train_all_models():
    products = get_product_list_json()

    for product in products:
        train_model(
            product["trainModelPath"],
            product["decisionColumn"],
            product["testModelPath"],
            product["name"],
            product["encodedColumns"],
            product["modelFlag"]
        )
    products = get_product_list_json()
    return jsonify(products), 200


@product_bp.route('/get-product', methods=['GET'])
def get_product_by_id():
    product_id = request.args.get('id')
    if not product_id:
        return jsonify({"error": "ID query parameter is required"}), 400
    product = get_product_json(int(product_id))
    return jsonify(product), 200


@product_bp.route('/get-correlation-matrix-image-df', methods=['POST'])
def get_correlation_plot_image_df():
    data = request.json
    try:
        model_name = data["model_name"]
        trainset_fileName = data["train_file_name"]
        file_list = os.listdir(commons_train_set_file_path)
        file_path = ""
        for val in file_list:
            if trainset_fileName in val:
                file_path = os.path.join(commons_train_set_file_path, val)
                break;

        if file_path == "":
            return jsonify({"error": "File not found"}), 500
        else:
            img = get_correlation_matrix_image(file_path, model_name)
            return send_file(img, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e.__cause__)}), 500


@product_bp.route('/get-correlation-matrix-image', methods=['GET'])
def get_correlation_plot_image():
    product_id = request.args.get('id')
    if not product_id:
        return jsonify({"error": "ID query parameter is required"}), 400
    product = get_product_json(int(product_id))
    img = get_correlation_matrix_image(product["trainModelPath"], product["name"])
    return send_file(img, mimetype='image/png')

@product_bp.route('/get-correlation-matrix-info', methods=['POST'])
def get_correlation_plot_text():
    data = request.json
    trainset_name = data['train_set_name']
    model_name = data['model_name']
    columns_to_encode = data['columns_to_encode']
    result_column = data['result_column']
    number_of_train_data = data['number_of_train_data']

    # Check if any of the required parameters are missing
    if not all([trainset_name, model_name, result_column, number_of_train_data]):
        return jsonify({"error": "improper request"}), 400

    corr_mat = get_correlation_matrix(trainset_name, model_name, columns_to_encode, result_column, number_of_train_data)
    return jsonify({"corr_mat": str(corr_mat)}), 200



# Register the blueprint with a URL prefix
app.register_blueprint(product_bp, url_prefix=url_prefix)


@app.route('/health', methods=['GET'])
def health_check():
    return 'Server is up and running!', 200


if __name__ == '__main__':
    app.run(debug=True, port=7654, host="0.0.0.0")
