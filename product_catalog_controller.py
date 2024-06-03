from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS
from json_operations import get_product_list_json


url_prefix = '/api'

app = Flask(__name__)

CORS(app)

# Define the blueprint
product_bp = Blueprint('product_bp', __name__)

@product_bp.route('/train-model', methods=['POST'])
def chat_single():
    data = request.json
    print(data)
    return jsonify({'output': "output"}), 200

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
    app.run(debug=True)
