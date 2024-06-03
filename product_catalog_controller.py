from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS
from flask_restful import Resource, Api
from json_operations import get_product_list_json


app = Flask(__name__)

CORS(app)

@app.route('/health', methods=['GET'])
def health_check():
    return 'Server is up and running!', 200

@app.route('/train-model', methods=['POST'])
def chat_single():
    data = request.json
    print(data)

    return jsonify({'output': "output"}), 200

@app.route('/get-all-products', methods=['GET'])
def get_all_products():
    products = get_product_list_json()
    return jsonify(products), 200


product_bp = Blueprint('product_bp', __name__)
app.register_blueprint(product_bp, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=True)
