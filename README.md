# Predictive Transaction Intelligence

A comprehensive fraud detection system built using Machine Learning and deployed with Streamlit.  
This project allows users to upload transaction datasets, score them using a trained model, and analyze fraud risk.

## 🚀 Features
- Upload CSV data and auto-normalize columns  
- Fraud scoring using ML model (Logistic Regression / Random Forest / LightGBM)  
- Adjustable fraud threshold and review band  
- Intelligent explanations using LLM (fallback rule-based)  
- Dashboard with insights  
- Modular code structure (`src/`)  

## 📂 Project Structure
```
predictive-tx-intel/
│── app_streamlit.py
│── requirements.txt
│── src/
│   ├── llm_explainer.py
│   ├── preprocess_and_train.ipynb
│   ├── data_split.ipynb
│── artifacts/   (stored externally)
```

## 📦 Artifacts Notice
Model files are **NOT included in GitHub** due to size limits.  
Store them externally (e.g., Google Drive) and reference them in your repo.
## 📁 Model Artifacts (Download)

Because GitHub limits file sizes, the ML model files are stored on Google Drive.

🔗 **Download Artifacts Here:**  
https://drive.google.com/file/d/1wqWKyfLCgjcKj4VK-8Sc5rb0Gh61PgSC/view?usp=sharing

Example structure:
```
artifacts/
  ├── fraud_model.joblib
  ├── label_encoders.joblib
  ├── metadata.joblib
```

## 🛠 Installation
```
git clone https://github.com/<yourname>/predictive-tx-intel.git
cd predictive-tx-intel
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Run the App
```
streamlit run app_streamlit.py
```

## 📄 License
This project is licensed under the MIT License.  
See `LICENSE` for details.
