from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

try:
    model = joblib.load('pokemon_model.pkl')
except Exception as e:
    model = None

@app.route('/')
def home():
    return "API Pokemon Lendário"

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'Modelo não foi carregado'}), 500

    try:
        data = request.get_json()
        
        valores_entrada = [
            float(data.get('hp')),
            float(data.get('attack')),
            float(data.get('defense')),
            float(data.get('sp_atk')),
            float(data.get('sp_def')),
            float(data.get('speed')),
            float(data.get('generation'))
        ]

        predicao = model.predict([valores_entrada])[0]
        
        resultado = "LENDÁRIO" if predicao else "Comum"
        
        return jsonify({'resultado': resultado})

    except Exception as e:
        return jsonify({'erro': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)