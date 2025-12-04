import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)

try:
    model = pickle.load(open("model.pkl", "rb"))
    names = pickle.load(open("names.pkl", "rb"))
    print("Modelos carregados com sucesso!")
except Exception as e:
    print(f"ERRO CRÍTICO ao carregar modelos: {e}")
    model = None
    names = None

@app.route("/")
def home():
    return "API ITS WORKS!"

@app.route("/api/predict", methods=["POST"])
def results():
    if not model or not names:
        return jsonify({"erro": "O modelo não foi carregado corretamente no servidor."}), 500

    try:
        data = request.get_json(force=True)
        
        features = np.array(list(data.values())).reshape(1, -1)
        
        pred = model.predict(features)
        
        output = names[pred[0]]
        
        return jsonify(output)
    
    except Exception as e:
        return jsonify({"erro": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)