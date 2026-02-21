# 📩 Email / SMS Spam Classifier  

An interactive web application that detects spam messages in real time using **Machine Learning and Natural Language Processing (NLP)**.

The system follows a **Layered Architecture (Client-Server Model)** with clear separation between UI, application logic, ML inference, and deployment layers to ensure maintainability and scalability.

---

# 🚀 Features  

- Detects spam messages using **Multinomial Naive Bayes**
- **TF-IDF Vectorization** for feature extraction
- NLP preprocessing pipeline:
  - Tokenization  
  - Stopword removal  
  - Stemming  
- Confidence score with probability display
- Interactive UI built using **Streamlit**
- Clean and consistent user interface
- Dockerized deployment
- Hosted on **Render**

---

# 🏗️ Software Design  

The system is designed using strong software engineering principles:

- ✅ High Cohesion  
- ✅ Low Coupling  
- ✅ Modularity  
- ✅ Abstraction  
- ✅ Maintainability  

## 🧱 Architecture Overview  

The application consists of four main layers:

### 1️⃣ Presentation Layer  
- Streamlit Web UI  

### 2️⃣ Application Layer  
- Input validation  
- Text preprocessing  
- Controller logic  

### 3️⃣ ML Inference Layer  
- TF-IDF Vectorizer  
- Naive Bayes Model  
- `model.pkl`  
- `vectorizer.pkl`  

### 4️⃣ Deployment Layer  
- Docker Container  
- Render / Localhost  

---

## 📊 Architecture Diagram  

Editable Draw.io file and PNG export are available in:

/design/architecture.drawio

/design/architecture.png



---

## 🔄 Data Flow Diagram  

The system processes input using the following pipeline:

User Input  
→ Input Validation  
→ Text Preprocessing  
→ TF-IDF Vectorization  
→ Model Prediction  
→ Confidence Score Calculation  
→ Result Display  

Data flow diagram available in:

/design/data_flow.png




---

# 📂 Project Structure  

.

├── app.py                  # Streamlit web application

├── model.pkl               # Trained ML model

├── vectorizer.pkl          # TF-IDF vectorizer

├── requirements.txt        # Python dependencies

├── Dockerfile              # Container setup

├── render.yaml             # Render configuration

├── README.md               # Project documentation

└── design/

├── architecture.drawio # Editable architecture diagram

├── architecture.png    # Architecture export

├── data_flow.png       # Data flow diagram

└── ui-screens/         # Figma screen exports



---

# 🛠️ Tech Stack  

- **Python 3.10**
- **scikit-learn**
- **NLTK**
- **Streamlit**
- **Docker**
- **Render**

---

# ⚡ How to Run Locally  

```bash
# Clone the repository
git clone [https://github.com/yourusername/spam-classifier.git](https://github.com/yourusername/spam-classifier.git)
cd spam-classifier

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```
