from flask import Flask, request, jsonify
import os
import openai
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd

app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": "http://localhost:3000"}})

# Load the API key from an environment variable
glhf_api_key = os.getenv("GLHF_API_KEY")
if not glhf_api_key:
    raise ValueError("GLHF_API_KEY environment variable not set")

# Initialize the OpenAI client with the specified base URL
client = openai.OpenAI(api_key=glhf_api_key, base_url="https://glhf.chat/api/openai/v1")

# Initialize the sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_text(text: str):
    return ' '.join(text.split())

# Global variables for the knowledge base
knowledge_base = []
kb_embeddings = model.encode(knowledge_base, convert_to_numpy=True)


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return ' '.join([para.text for para in doc.paragraphs])

def initialize_knowledge_base():
    global knowledge_base, kb_embeddings
    document_paths = {
        'csv': [
            'clase_de_med_RA.csv',
            'teste_comorb_rin_alg.csv',
            'teste_comorb_rin_alt_no_diacritics.csv',
            'teste_diag_dif_rinita_alg.csv',
            'teste_diag_dif_rinita_alg_no_diacritics.csv',
            'tratam_RA_persistenta.csv',
            'tratam_RA_interm.csv',
            'tratam_conj.csv'
        ],
        'json': [
            'diagnosticul_RA.json'
        ],
    }

        # Read CSV files
    for csv_file in document_paths['csv']:
        try:
            import pandas as pd
            df = pd.read_csv(csv_file)
            text = df.to_string()
            knowledge_base.append({"topic": "medical_data", "content": clean_text(text)})
        except Exception as e:
            print(f"Error reading CSV file {csv_file}: {e}")

    # Read JSON files
    for json_file in document_paths['json']:
        try:
            import json
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                text = json.dumps(data, ensure_ascii=False, indent=2)
                knowledge_base.append({"topic": "medical_data", "content": clean_text(text)})
        except Exception as e:
            print(f"Error reading JSON file {json_file}: {e}")

    kb_embeddings = model.encode([entry["content"] for entry in knowledge_base], convert_to_numpy=True)

initialize_knowledge_base()

def retrieve_documents(query, topic=None):
    query_embedding = model.encode(query, convert_to_numpy=True).reshape(1, -1)
    if topic:
        filtered_kb = [entry["content"] for entry in knowledge_base if entry["topic"] == topic]
        filtered_embeddings = model.encode(filtered_kb, convert_to_numpy=True)
        similarities = cosine_similarity(query_embedding, filtered_embeddings)
    else:
        similarities = cosine_similarity(query_embedding, kb_embeddings)

    top_indices = np.argsort(similarities[0])[-5:][::-1]
    matching_docs = [knowledge_base[idx]["content"] for idx in top_indices]
    return matching_docs



# Dictionary to store conversation history for each user
conversation_histories = {}


def calculate_patient_score(contexts):
    """
    Improved patient scoring logic based on retrieved contexts.
    """
    symptoms = ['sneezing', 'runny nose', 'congestion']
    severity_keywords = ['severe', 'chronic', 'persistent']
    score = 0

    for context in contexts:
        for symptom in symptoms:
            if symptom in context.lower():
                score += 1
        for keyword in severity_keywords:
            if keyword in context.lower():
                score += 2

    return {"severity_score": score}


def generate_response(conversation_history, contexts, language="en"):
    """
    Generate a response by incorporating retrieved contexts with conversation history.
    Language parameter ensures responses are generated in the desired language.
    """
    conversation_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
    contexts_str = "\n".join(contexts)

    prompt = f"""
    You are a helpful medical assistant with access to the following knowledge base:
    {contexts_str}

    {conversation_str}
    Assistant (in {language}):
    """
    try:
        response = client.chat.completions.create(
            model="hf:meta-llama/Meta-Llama-3.1-405B-Instruct",
            messages=[{"role": "system", "content": prompt}],
            stream=False,
        )
        assistant_response = response.choices[0].message.content.strip()
        return assistant_response
    except Exception as e:
        return f"Error generating response: {str(e)}"

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    if not data or 'question' not in data:
        return jsonify({'error': 'Invalid input: question is required'}), 400

    user_id = data.get('user_id', 'default_user')  # Use a unique ID to track conversation
    question = data['question']
    language = data.get('language', 'en')

    if user_id not in conversation_histories:
        conversation_histories[user_id] = [
            {"role": "system", "content": "You are a medical assistant. Collect the user's symptoms to help determine their respiratory condition and provide a possible diagnosis."}
        ]
    # Add the user's question to their conversation history
    conversation_histories[user_id].append({"role": "user", "content": question})


    # Logic to ask follow-up questions
    follow_up_questions = {
        "cough": "Do you have a dry or productive cough?",
        "fever": "Do you have a fever? If so, how high?",
        "shortness_of_breath": "Are you experiencing shortness of breath, especially during physical activity?",
        "sore_throat": "Do you have a sore throat? Is it accompanied by difficulty swallowing?",
        "nasal_congestion": "Are you experiencing nasal congestion or runny nose?"
    }

    last_user_response = question.lower()
    next_question = None
    for keyword, follow_up in follow_up_questions.items():
        if keyword in last_user_response:
            next_question = follow_up
            break

        # If no follow-up question matches, ask a general health question
    if not next_question:
        next_question = "Can you describe any other symptoms you're experiencing?"

    contexts = retrieve_documents(question)

    assistant_response = generate_response(conversation_histories[user_id], contexts)
    # Add the follow-up question to the assistant's response# If the conversation includes enough data, provide a diagnosis
    collected_symptoms = " ".join([msg['content'] for msg in conversation_histories[user_id] if msg['role'] == 'user']).lower()
    if "cough" in collected_symptoms and "nasal congestion" in collected_symptoms:
        assistant_response += " Based on your symptoms, you may have allergic rhinitis. However, I recommend consulting a medical professional for an accurate diagnosis."
    elif "fever" in collected_symptoms and "sore throat" in collected_symptoms:
        assistant_response += " Based on your symptoms, you may have a bacterial infection like strep throat. Please consult a medical professional for further evaluation."

    # Append the assistant's response and follow-up question to the conversation history
    conversation_histories[user_id].append({"role": "assistant", "content": assistant_response + " " + next_question})

    # Remove the user's last input from the conversation history
    conversation_histories[user_id] = [msg for msg in conversation_histories[user_id] if not (msg['role'] == 'user' and msg['content'] == question)]

    return jsonify({
        'question': question,
        'response': assistant_response + " " + next_question,
        'conversation_history': conversation_histories[user_id]
    })


if __name__ == '__main__':
    app.run(debug=True)