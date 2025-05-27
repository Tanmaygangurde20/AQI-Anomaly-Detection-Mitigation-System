from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
import json

app = Flask(__name__)

# Load the models
try:
    iso_forest = joblib.load('iso_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    print("Warning: Model files not found")

# Initialize LLM
MODEL = "gemma2:2b"
try:
    model = ChatOllama(model=MODEL, temperature=0)
except Exception as e:
    print(f"Warning: Could not initialize LLM: {e}")
    model = None

# Load and initialize GRAP document for RAG
try:
    PDF_FILE = "GRAP.pdf"
    loader = PyPDFLoader(PDF_FILE)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    chunks = splitter.split_documents(pages)
    embeddings = OllamaEmbeddings(model=MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()
except Exception as e:
    print(f"Warning: Could not initialize RAG components: {e}")
    retriever = None

# Mitigation strategies function using the RAG approach
def get_mitigation_strategies(aqi_level):
    try:
        if retriever:
            retrieved_context = retriever.invoke(f"What mitigation strategies are recommended for AQI level {aqi_level}?")
            prompt_str = f"""
            You are an AI assistant designed to recommend mitigation strategies based on air quality index (AQI) levels using the Graded Response Action Plan (GRAP).

Based on the given AQI level, provide the appropriate mitigation actions as per GRAP:

- **Stage I (Poor)**: 201-300
- **Stage II (Very Poor)**: 301-400
- **Stage III (Severe)**: 401-450
- **Stage IV (Severe+)**: > 450

If the AQI level is not in these ranges, respond with "No action needed."

Context: {retrieved_context}

- **Vehicle Actions**: Retrieve recommendations related to vehicle restrictions during high pollution. 
- **Industry Actions**: Retrieve actions for industries to minimize pollution.

Example Response Format:
Alert: Due to high pollution levels, avoid using your vehicle.

**Vehicle Type**: Petrol Car  
**Action Required**: Non-compliance may result in a fine.

Please retrieve the relevant actions from the document and provide them in a list of 3-5 key recommendations. Include any fines or penalties associated with non-compliance.
"""
            
        else:
            prompt_str = f"""
            You are an AI assistant providing mitigation strategies based on air quality index (AQI) levels.
            Based on the AQI level of {aqi_level}, recommend specific actions that should be taken.
            """
        
        response = model.invoke(prompt_str)
        return str(response)
    except Exception as e:
        return "Error generating mitigation strategies. Please try again later."


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features in correct order
        features = ['AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity', 'WindSpeed']
        input_data = [float(data[f]) for f in features]
        
        # Prepare input for model
        input_array = np.array(input_data).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        
        # Make prediction
        prediction = iso_forest.predict(scaled_input)
        anomaly_score = iso_forest.score_samples(scaled_input)
        result = "Normal" if prediction[0] == 1 else "Anomalous"
        
        # Generate LIME explanation
        n_samples = 1000
        synthetic_samples = scaled_input + np.random.normal(0, 0.5, (n_samples, len(features)))
        synthetic_predictions = iso_forest.predict(synthetic_samples)
        
        surrogate_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        surrogate_clf.fit(synthetic_samples, synthetic_predictions)
        
        explainer = LimeTabularExplainer(
            synthetic_samples,
            feature_names=features,
            class_names=['Anomaly', 'Normal'],
            mode='classification',
            random_state=42
        )
        
        explanation = explainer.explain_instance(
            scaled_input[0],
            surrogate_clf.predict_proba,
            num_features=5
        )
        
        # Get feature importance
        feature_importance = []
        for feature, weight in explanation.as_list():
            feature_importance.append({
                'feature': feature.replace(" ", ""),
                'weight': round(weight, 4)
            })
        
        # Get mitigation strategies if AQI is provided
        mitigation_strategies = get_mitigation_strategies(data['AQI'])
        
        return jsonify({
            'status': 'success',
            'prediction': result,
            'anomaly_score': round(float(anomaly_score[0]), 3),
            'feature_importance': feature_importance,
            'mitigation_strategies': mitigation_strategies,
            'input_summary': dict(zip(features, input_data))
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT not set
    app.run(debug=True, host='0.0.0.0', port=port)
