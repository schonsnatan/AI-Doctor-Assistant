# 🤖 Doctor Assistant AI

An AI-powered clinical assistant built with **Streamlit**, **LLaMA 3 (via Groq)**, and **vector search (RAG)**.  
This app simulates a medical assistant that suggests **medications** and **exams** based on patient data — while respecting allergies and past reactions.

![Alt Text](doctor-assistant.gif)

## 🚀 Features

- 🔎 Intelligent suggestions of medications and lab exams
- ✅ Allergy and adverse reaction filtering logic
- 🧠 LLM reasoning with Groq (LLaMA 3 70B)
- 🔍 Vector search with FAISS + HuggingFace embeddings
- 📦 Modular architecture (strategies, services, core models)
- 🖥️ Interactive UI built with Streamlit



## 🧠 How it Works

1. **User inputs** patient data through a simple and clean Streamlit interface.
2. The app sends the data to an **LLM hosted on Groq**, which generates medical search terms (e.g., drug classes, active ingredients).
3. These terms are used in a **vector search (RAG)** over a local database of medications.
4. Documents with **forbidden substances** (e.g., based on allergies) are **filtered out**.
5. The LLM makes a final recommendation based on the patient’s context and the safe options retrieved.



## 🗂️ Project Structure

```
doctor_assistant/
├── app.py                  # UI (Streamlit)
├── core/                   # Patient data model and orchestrator
│   ├── patient.py
│   └── suggestion_engine.py
├── strategies/             # Strategy design pattern (meds, exams)
│   ├── base_strategy.py
│   ├── medicine_strategy.py
│   └── exam_strategy.py
├── services/               # LLM and RAG services
│   ├── llm_service.py
│   └── rag_service.py
├── data/                   # Medication dataset + vectorstore
```



## 📸 UI Preview

<img width="1266" height="568" alt="image" src="https://github.com/user-attachments/assets/526514c9-1770-4a52-976e-ef6a83a490e7" />

## 🧠 AI Response

<img width="1232" height="587" alt="image" src="https://github.com/user-attachments/assets/1e2ddfb7-0ce8-4a9e-a4f4-abce6d46e78a" />

## 🚀 Setup (using Poetry)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/doctor-assistant.git
   cd doctor-assistant
   ```
2. Install the dependencies:

  ```bash
   poetry install
   ```

3. Activate the virtual enviroment:

 ```bash
   poetry shell
   ```

4. ✅ Make sure to set up your .env file with your GROQ API key

```python
GROQ_API_KEY=your_groq_api_key_here
```

5. Run the streamlit app:

```bash
   streamlit run app.py
   ```  

## 🤝 Disclaimer

This project is a **proof of concept** for educational and prototyping purposes only.  
It is **not intended for real medical use**. Always consult licensed medical professionals for diagnosis or treatment.
