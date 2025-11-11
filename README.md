# 🎬 Movie Review Similarity Finder

An AI-powered web application that finds similar movie reviews using semantic similarity. Compare your thoughts with existing reviews for Fight Club and Interstellar.

## Features

- **Dual Interface**: FastAPI backend + Streamlit frontend
- **AI-Powered Matching**: Uses Groq LLM for semantic understanding
- **Multiple Search Modes**: TF-IDF similarity or AI-enhanced filtering
- **Two Movie Databases**: Fight Club and Interstellar review collections
- **Smart Filtering**: Get fewer but more relevant results with AI

## Installation

```bash
git clone <your-repo-url>
cd movie-review-similarity-app
pip install -r requirements.txt
```

Run Backend (Terminal 1):
-------------------------

` uvicorn main:app --reload --host 0.0.0.0 --port 8000   `

Run Frontend (Terminal 2):
--------------------------

`   streamlit run streamlit_app.py   `

Access
------

*   **Backend API**: [http://localhost:8000](http://localhost:8000/)
    
*   **Frontend App**: [http://localhost:8501](http://localhost:8501/)
    
*   **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
    

API Endpoints
-------------

*   POST /find-similar-reviews - Find similar reviews
    
*   GET /movies - List available movies
    
*   GET /health - Health check
    

Technologies
------------

*   **Backend**: FastAPI, Scikit-learn, Pandas
    
*   **Frontend**: Streamlit
    
*   **AI**: Groq LLM, LangChain
    
*   **Similarity**: TF-IDF + Cosine Similarity
