from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restful import Resource, Api

model_path = "model/"

app = Flask(__name__)
api = Api(app, "/api")

CORS(app)

@app.route('/health', methods=['GET'])
def health_check():
    return 'Server is up and running!', 200

@app.route('/train-model', methods=['POST'])
def chat_single():
    data = request.json
    print(data)

    return jsonify({'output': "output"}), 200

if __name__ == '__main__':
    app.run(debug=True)
