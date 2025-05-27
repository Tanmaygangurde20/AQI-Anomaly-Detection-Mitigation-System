Here is a detailed and professional `README.md` for your **Air Quality Anomaly Detection & Mitigation System** project, including sections for overview, features, setup, usage, tech stack, and contribution guidelines:

---

````markdown
# 🌫️ Air Quality Anomaly Detection & GRAP-based Mitigation System

An intelligent AI-powered Flask application to detect anomalies in air quality data and recommend real-time **mitigation strategies** using GRAP (Graded Response Action Plan) guidelines. Built with **Machine Learning**, **LLMs**, and **LIME explanations** to promote transparency and actionable insights for public health and climate control efforts.

---

## 📌 Project Overview

The system does **two major things**:

1. **Detects Anomalous AQI patterns** using Isolation Forest.
2. **Provides GRAP-compliant Mitigation Strategies** using a **Retrieval-Augmented Generation (RAG)** pipeline powered by **LLMs (Gemma2:2b)**.

Additionally:
- 🧠 Model explainability is integrated using **LIME** and **RandomForest** surrogate models.
- 🗂️ Suggestions are extracted from the **GRAP PDF document** using semantic search and summarization.

---

## 🎯 Key Features

### ✅ Anomaly Detection
- Uses **Isolation Forest** to detect abnormal pollution levels.
- Inputs: AQI, PM10, PM2.5, NO2, SO2, O3, Temperature, Humidity, WindSpeed.

### ✅ Mitigation Strategies via RAG
- Retrieves actions from the **GRAP document** using vector search (FAISS + Ollama Embeddings).
- Generates contextual actions using **LangChain + ChatOllama**.
- Adapts responses according to AQI level:
  - Stage I: Poor (201–300)
  - Stage II: Very Poor (301–400)
  - Stage III: Severe (401–450)
  - Stage IV: Severe+ (>450)

### ✅ Explainable AI (LIME)
- Provides **feature-level reasoning** for anomaly predictions.
- Visual summary of top 5 most influential features.

---

## 📊 Input Fields

| Feature        | Description                       |
|----------------|------------------------------------|
| AQI            | Air Quality Index                  |
| PM10           | Particulate Matter ≤10 microns     |
| PM2.5          | Particulate Matter ≤2.5 microns    |
| NO2            | Nitrogen Dioxide                   |
| SO2            | Sulphur Dioxide                    |
| O3             | Ozone                              |
| Temperature    | Ambient Temperature (°C)           |
| Humidity       | Relative Humidity (%)              |
| WindSpeed      | Wind Speed (km/h)                  |

---

## 🚀 Tech Stack

| Category               | Technology / Tool                |
|------------------------|----------------------------------|
| Backend Framework      | Flask                            |
| ML/AI Models           | Isolation Forest, Random Forest, LIME |
| RAG/LLM Layer          | LangChain, ChatOllama, Ollama Embeddings |
| Explainability         | LIME (LimeTabularExplainer)      |
| Document Processing    | FAISS, PyPDFLoader, RecursiveTextSplitter |
| Language Models        | Gemma2:2b                        |
| Frontend               | HTML (via Flask templates)       |

---

## ⚙️ Setup Instructions

1. **Clone this repository**

```bash
git clone https://github.com/your-username/air-quality-grap-system.git
cd air-quality-grap-system
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Place your model files**

Ensure you have:

* `iso_forest_model.pkl`
* `scaler.pkl`
* `GRAP.pdf`

4. **Run the app**

```bash
python app.py
```

Then, visit `http://127.0.0.1:5000` in your browser.

---

## 🧪 API Usage (Optional)

`POST /predict`

### Sample JSON:

```json
{
  "AQI": 345,
  "PM10": 200,
  "PM2_5": 125,
  "NO2": 90,
  "SO2": 30,
  "O3": 50,
  "Temperature": 24,
  "Humidity": 55,
  "WindSpeed": 10
}
```

### Response:

```json
{
  "prediction": "Anomalous",
  "anomaly_score": -0.32,
  "feature_importance": [
    {"feature": "PM2_5", "weight": 0.45},
    {"feature": "NO2", "weight": 0.35}
  ],
  "mitigation_strategies": "Limit diesel vehicles, shut down brick kilns...",
  "input_summary": { ... }
}
```

---

## 🛡️ AQI Stage Classification (GRAP)

| AQI Level | Stage     | Suggested Action                          |
| --------- | --------- | ----------------------------------------- |
| 201–300   | Stage I   | Public awareness, water sprinkling        |
| 301–400   | Stage II  | Halt diesel generators, parking fee hikes |
| 401–450   | Stage III | Closure of brick kilns, hot mix plants    |
| > 450     | Stage IV  | Closure of schools, construction ban      |

---

## 💡 Ideas for Future Work

* 📡 Live AQI data integration from government APIs
* 📱 Mobile UI with push notifications
* 🔗 GRAP compliance audit tool for industries
* 📊 Admin dashboard for mitigation metrics

---



## 📬 Contact

* **Email:** [tanmaygangurde10@gmail.com](mailto:tanmaygangurde10@gmail.com)
* **LinkedIn:** [Tanmay Gangurde](https://www.linkedin.com/in/tanmay-gangurde-112856265)

---

> 🔔 *Built to bridge data and decisions for a cleaner tomorrow.*
