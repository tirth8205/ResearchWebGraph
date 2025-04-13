# ResearchWebGraph

ResearchWebGraph is an AI-powered research assistant that helps users explore, analyze, and understand academic papers. The application combines advanced AI technologies like knowledge graphs, vector search, and large language models (LLMs) to deliver insights from research papers.

---

## Features

### 📚 Paper Search
- Search for research papers on arXiv using keywords and filters.
- Upload PDF documents to extract text and analyze their content.

### 🕸️ Knowledge Graph
- Build interactive knowledge graphs to visualize entities, concepts, and relationships extracted from research papers.
- Explore connections between entities like people, organizations, and academic terms.

### 🤖 Query Assistant
- Ask questions about selected papers and receive AI-generated answers with citations.
- Compare methodologies, summarize findings, and explore relationships between papers.

---

## Technology Stack

### Backend
- **FastAPI**: Backend framework for handling API requests.
- **Qdrant**: Vector database for storing embeddings and performing semantic search.
- **SentenceTransformers**: For generating embeddings from text.
- **PyPDF2**: For extracting text from PDF documents.
- **spaCy & NLTK**: For natural language processing tasks like entity extraction.

### Frontend
- **Streamlit**: Interactive web interface for users.
- **streamlit-option-menu**: For navigation between pages.
- **Requests**: For communicating with the backend API.

### LLMs
- **Groq API**: Access to powerful large language models (e.g., Llama 3) for answering user queries.

---

## Installation

### Prerequisites
1. Python 3.9 or higher installed on your system.
2. Docker installed (for running Qdrant as a vector database).
3. A valid Groq API key for accessing LLMs.

---

### Backend Setup

1. Navigate to the `backend` directory:
   ```
   cd backend
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Start the Qdrant vector database:
   ```
   docker run -p 6333:6333 qdrant/qdrant
   ```

4. Run the FastAPI backend:
   ```
   uvicorn app.main:app --reload
   ```

---

### Frontend Setup

1. Navigate to the `frontend` directory:
   ```
   cd frontend
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Start the Streamlit application:
   ```
   streamlit run app.py
   ```

---

## Usage

1. Open the Streamlit frontend in your browser (e.g., `http://localhost:8501`).
2. Configure your Groq API key in the sidebar.
3. Use the "Research Papers" tab to search for papers or upload PDF documents.
4. Select papers to analyze and build a knowledge graph in the "Knowledge Graph" tab.
5. Use the "Query Assistant" tab to ask questions about your selected papers.

---

## Project Structure

```
ResearchWebGraph/
├── backend/
│   ├── app/
│   │   ├── routers/           # API endpoints (papers, query, graph)
│   │   ├── services/          # Core business logic (processing, graph building)
│   │   ├── utils/             # Helper utilities (PDF processing, NLP)
│   │   ├── models/            # Pydantic schemas for request/response validation
│   │   └── main.py            # FastAPI entry point
│   ├── requirements.txt       # Backend dependencies
│   └── .env                   # Environment variables for backend configuration
├── frontend/
│   ├── app.py                 # Streamlit entry point
│   ├── components/            # Reusable UI components (API client, sidebar)
│   ├── pages/                 # Multi-page Streamlit app (papers, graph, assistant)
│   ├── requirements.txt       # Frontend dependencies
│   └── .env                   # Environment variables for frontend configuration
└── README.md                  # Project overview and setup instructions
```

---

## Environment Variables

### Backend (.env)
```
GROQ_API_KEY=your-groq-api-key-here
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=research_papers
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Frontend (.env)
```
BACKEND_URL=http://localhost:8000
GROQ_API_KEY=your-groq-api-key-here
```

---

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your changes.

---

## License

This project is licensed under the MIT License. See `LICENSE` for more details.

