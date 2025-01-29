# Allergies Helper Assistant

The **Allergies Helper Assistant** is a web application designed to help users identify potential respiratory issues based on their symptoms. It dynamically interacts with users, retrieves relevant medical knowledge, and generates AI-powered responses to provide guidance.

---

## Features

- **Interactive User Interface**:
  - Chat-like interface where users can input their symptoms.
  - Dynamic follow-up questions to gather detailed information.

- **Backend with Flask**:
  - Tracks conversation history for each user.
  - Integrates with a structured knowledge base to retrieve relevant contexts.
  - Generates responses using the OpenAI API.

- **AI-Powered Responses**:
  - Leverages the OpenAI API to provide contextual and detailed medical guidance.
  - Dynamically generates follow-up questions and possible diagnoses.

- **Knowledge Base Integration**:
  - Extracts data from medical documents (PDFs, DOCX).
  - Provides up-to-date resources for symptom analysis and response generation.

- **Multilingual Support**:
  - Responds in the user's preferred language (e.g., English, Romanian).

- **Frontend (React)**:
  - Clean and responsive UI for smooth interaction.
  - Displays scrollable responses for detailed outputs.



## Requirements

- **Frontend**:
  - React.js
  - Axios for API communication.

- **Backend**:
  - Flask
  - Flask-CORS for cross-origin resource sharing.
  - OpenAI API for AI-powered responses.
  - Sentence Transformers for knowledge base search.

- **Knowledge Base**:
  - Extracts and processes medical data from PDFs and DOCX files.



## Installation and Setup

### Prerequisites
- Node.js
- Python 3.8 or later
- pip (Python package manager)

### Clone the Repository
```bash
git clone https://github.com/raulradulescu/RAG-IP-application.git
cd RAG-IP-application
```


## Backend Setup

1. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set your OpenAI API key as an environment variable:

    ```bash
    export GLHF_API_KEY=<your_openai_api_key>
    # On Windows:
    set GLHF_API_KEY=<your_openai_api_key>
    ```

4. Run the Flask backend:

    ```bash
    python app.py
    ```



## Frontend Setup

1. Navigate to the `rag-webapp` directory:

    ```bash
    cd rag-webapp
    ```

2. Install dependencies:

    ```bash
    npm install
    ```

3. Start the React development server:

    ```bash
    npm start
    ```



## Usage

1. Open the frontend in your browser (`http://localhost:3000`).
2. Enter your symptoms into the input field.
3. View the AI-generated responses and follow-up questions.
4. Get a possible diagnosis and guidance based on your symptoms.
