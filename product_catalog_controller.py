from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS
from json_operations import get_product_list_json
from train_model import train_model


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
        results = train_model(data["train_file_path"], data["decision_column"], data["test_file_path"], data["model_name"], data["model_flag"])

        # if isinstance(results, tuple):
        #     # If results is a tuple, convert it to a JSON response
        #     response = {result if (i + 1) % 2 != 0 else f"result_{i+1}" : result for i, result in enumerate(results)}
        # else:
        #     # If results is a single value, return it directly
        #     response = {"accuracy": results}

        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@product_bp.route('/get-all-products', methods=['GET'])
def get_all_products():
    products = get_product_list_json()
    return jsonify(products), 200

# Register the blueprint with a URL prefix
app.register_blueprint(product_bp, url_prefix=url_prefix)

@app.route('/health', methods=['GET'])
def health_check():
    return 'Server is up and running!', 200

if __name__ == '__main__':
    app.run(debug=True, port=8080)
