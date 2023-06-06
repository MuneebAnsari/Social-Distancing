import torch
import utils
from flask import Flask, jsonify, request, make_response
from models import FaceMaskNet


app = Flask(__name__)

model = FaceMaskNet()
model.load_state_dict(torch.load('facemasknet.pt',  map_location=torch.device('cpu')))
model.eval()

@app.route('/')
def index():
    return jsonify({"message": "FaceMaskNet API"}), 200


@app.route('/api/v1/predict', methods=["POST"]) 
def predict():
    try:
        if not request.files or not request.files['file']:
            return jsonify({"message": "Bad Request 'file' not provided in request form-data"}), 400
        
        req_file = request.files['file'].read()
        if not utils.is_valid_image(req_file):
            return jsonify({"message": "Bad Request image" }), 400
        
        image = req_file
        X = utils.build_input_tensor(image)
        pred = utils.predict(X, model)
        featureMaps = utils.get_features(model, X)

        response_data = {
            "prediction": pred,
            "features": featureMaps
         }

        response = make_response(response_data, 200)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        print(e)
        return jsonify({"message": "Internal Server Error"}), 500
    

if __name__ == "__main__":
    app.run()