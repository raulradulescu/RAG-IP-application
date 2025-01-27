import React, { useState } from 'react';
import axios from 'axios';
import './App.css'; // Import the CSS file
import { v4 as uuidv4 } from 'uuid';

function App() {
  const [userId] = "abc123"/*useState(uuidv4())*/; // Hardcoded user ID for now
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState('');
  const [patientScore, setPatientScore] = useState(null); // State for patient score
  const [language, setLanguage] = useState('en');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
        const result = await axios.post('http://127.0.0.1:5000/ask', {
            user_id: userId,  // Pass user ID for conversation tracking
            question
        });
        setResponse(result.data.response);
    } catch (error) {
        console.error('Error fetching data:', error);
        if (error.response) {
            setError(`Server error: ${error.response.status} - ${error.response.data}`);
        } else if (error.request) {
            setError('No response received. Network error.');
        } else {
            setError(`Error: ${error.message}`);
        }
        setResponse('');
    } finally {
        setQuestion('');  // Clear the input field
        setLoading(false);
    }
};

  return (
    <div className="app-container">
      <header>
        <h1>Medical assitant specialized on respiratory issues</h1>
      </header>
      <main>
        <form onSubmit={handleSubmit} className="question-form">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question..."
            className="input-field"
          />
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="language-selector"
          >
            <option value="en">English</option>
            <option value="ro">Romanian</option>
          </select>
          <button type="submit" className="submit-button" disabled={loading}>
            {loading ? 'Loading...' : 'Ask'}
          </button>
        </form>
        {error && <div className="error-message">{error}</div>}
        {response && (
          <div className="response-container scrollable-response">
            <h2>Response:</h2>
            <p className="response-text">{response}</p>
            {patientScore && (
              <>
                <h3>Patient Score:</h3>
                <p>{`Severity Score: ${patientScore.severity_score || 'N/A'}`}</p>
              </>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
