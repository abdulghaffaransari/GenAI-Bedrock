
---

# **RAG-Driven RainAI**  
### **A Cutting-Edge Rainfall Prediction and Retrieval-Augmented Generation (RAG) System**  

---

## **Table of Contents**  
1. [Overview](#overview)  
2. [Key Features](#key-features)  
3. [Technologies Used](#technologies-used)  
4. [System Workflow](#system-workflow)  
5. [How to Set Up](#how-to-set-up)  
6. [Results](#results)  
7. [Future Enhancements](#future-enhancements)  
8. [Why RAG-Driven RainAI?](#why-rag-driven-rainai)  
9. [About the Developer](#about-the-developer)  

---

## **1. Overview**  

**RAG-Driven RainAI** is a **state-of-the-art Retrieval-Augmented Generation (RAG) system** powered by **Amazon Bedrock**, enabling intelligent question-answering and rainfall prediction.  

- **Rainfall Prediction**: Predict the amount of rainfall using historical weather datasets and advanced **machine learning models**.  
- **Document-Based QA**: Query large PDF documents to get intelligent answers, combining **vector-based retrieval** with **AI-powered generation**.  

---

## **2. Key Features**  

### 🌧️ **Rainfall Prediction**  
- Predict rainfall amounts (in mm) using 10 advanced machine learning regression models.  
- **Random Forest Regressor** achieved the highest accuracy, evaluated using:  
   - R² Score  
   - Mean Absolute Error (MAE)  
   - Root Mean Squared Error (RMSE).  

### 📄 **Document Querying System**  
- Upload **PDF documents** containing research or weather data.  
- Process PDFs to extract answers to user questions.  

### 🚀 **AI-Powered Answers**  
- Uses **Amazon Titan LLM** (via Amazon Bedrock) to generate contextual and accurate answers.  

### 🖥️ **Interactive Web Interface**  
- Powered by **Streamlit** for real-time interaction.  
- User-friendly features to store vectors and query documents seamlessly.  

---

## **3. Technologies Used**  

| **Category**               | **Tools/Technologies**                  |  
|----------------------------|-----------------------------------------|  
| **Language Models**        | Amazon Bedrock (Titan Embeddings, Titan Text) |  
| **Vector Search**          | FAISS (Facebook AI Similarity Search)  |  
| **Document Processing**    | PyPDF, LangChain                       |  
| **Web Framework**          | Streamlit                              |  
| **Backend & Cloud Services** | Boto3 (AWS SDK for Python)            |  
| **Machine Learning Models** | Random Forest, XGBoost, Linear Regression |  

---

## **4. System Workflow**  

The system consists of two core pipelines:  

### **A. Rainfall Prediction Pipeline**  
1. **Data Preprocessing**: Historical weather datasets are loaded and cleaned.  
2. **Model Training**: 10 supervised regression models are applied to predict rainfall amounts.  
3. **Model Evaluation**: Performance is evaluated using R² Score, MAE, and RMSE.  
4. **Prediction**: The best model (**Random Forest Regressor**) predicts rainfall values.  

### **B. Retrieval-Augmented Generation (RAG) Pipeline**  
1. **Document Ingestion**: PDFs are loaded using **PyPDF** and split into chunks using **LangChain's RecursiveCharacterTextSplitter**.  
2. **Embedding Creation**:  
   - Chunks are converted into embeddings using **Amazon Titan Embeddings**.  
   - Stored and indexed in a **FAISS** vector database.  
3. **Question-Answering**:  
   - User inputs a query.  
   - Similar chunks are retrieved using FAISS.  
   - The retrieved context is passed to **Amazon Titan LLM** to generate an accurate answer.  
4. **Output**: The final answer is displayed to the user.  

---

## **5. How to Set Up**  

### **Step 1: Clone the Repository**  
```bash
git clone https://github.com/abdulghaffaransari/RAG-Driven-RainAI.git
cd RAG-Driven-RainAI
```

### **Step 2: Install Dependencies**  
Ensure all required libraries are installed:  
```bash
pip install -r requirements.txt
```

### **Step 3: Run the Application**  
Launch the Streamlit web interface:  
```bash
streamlit run main.py
```

### **Step 4: Use the Application**  
1. **Upload PDF Files**: Click "Store Vector" in the sidebar to process PDF documents.  
2. **Ask a Question**: Input your query in the text box and click "Send".  
3. **Get Predictions**: View AI-generated answers and rainfall predictions.  

---

## **6. Results**  

### **Rainfall Prediction Results**  
The **Random Forest Regressor** achieved:  
- **R² Score**: 0.8699  
- **MAE**: 0.1944  
- **MSE**: 0.1263  
- **RMSE**: 0.3554  

### **Visual Results**  
The following visualization demonstrates the system's performance:  

![Predicted Results](https://github.com/abdulghaffaransari/RAG-Driven-RainAI/blob/main/Results/result%201.png)  

---

## **7. Future Enhancements**  
1. **Real-Time Weather Integration**: Add live APIs for real-time rainfall prediction.  
2. **Multi-Format Support**: Process additional file formats such as CSV, DOCX, and JSON.  
3. **Explainability**: Integrate tools like **SHAP** or **LIME** for model explainability.  
4. **Scalability**: Deploy the system using **AWS Lambda** or **Docker** for serverless scaling.  

---

## **8. Why RAG-Driven RainAI?**  
- Combines **machine learning**, **semantic search**, and **AI generation** for intelligent solutions.  
- Provides actionable insights for **agriculture, weather forecasting**, and **research industries**.  
- Built with scalable, production-grade technologies.  

---

## **9. About the Developer**  

**Abdul Ghaffar Ansari**  
- **AI & ML Engineer**  
- Expertise in Machine Learning, NLP, and AWS Cloud Services.  
- Passionate about solving real-world problems through data-driven AI solutions.  

**Contact Information**:  
- **Email**: [abdulghaffaransari9@gmail.com](mailto:abdulghaffaransari9@gmail.com)  
- **LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/abdulghaffaransari/)  
- **GitHub**: [GitHub Profile](https://github.com/abdulghaffaransari)  

---

## **Conclusion**  

**RAG-Driven RainAI** represents an innovative approach to solving critical challenges in rainfall prediction and intelligent document querying. Combining the power of **machine learning** with **retrieval-augmented generation**, this system provides precise and actionable insights, positioning itself as an impactful solution for **weather analysis** and **context-aware AI systems**.

---
