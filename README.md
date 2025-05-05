# ResearchWebGraph

**ResearchWebGraph** is an AI-powered research assistant that helps you explore and understand academic papers. It combines advanced technologies like knowledge graphs, vector search, and large language models (LLMs) to provide insights into research topics.

## Features

- **Search for Papers**: Retrieve research papers from arXiv based on keywords and filters.
- **Upload PDFs**: Upload your own research papers or documents for analysis.
- **Build Knowledge Graphs**: Extract entities and relationships from papers and visualize them as interactive graphs.
- **AI-Powered Query Assistant**: Ask questions about selected papers and get detailed answers with citations.


## Technology Stack

### Backend

- **FastAPI**: High-performance API framework.
- **Qdrant**: Vector database for semantic search.
- **SentenceTransformers**: Embedding model for generating vector representations of text.
- **PyPDF2 \& PDFMiner**: Tools for extracting text and metadata from PDFs.
- **spaCy \& NLTK**: NLP libraries for entity extraction and text processing.


### Frontend

- **Streamlit**: Interactive web interface for users.
- **streamlit-option-menu**: Navigation menu for multi-page apps.


### LLMs

- **Groq API**: Access to high-performance large language models like `llama-3.1-8b-instant` and `llama-3.3-70b-versatile`.


## Screenshots \& Demos

### Paper Search Interface

*Search for academic papers with filters for date and categories*

### Knowledge Graph Visualization

*Interactive knowledge graph showing relationships between research concepts*

### Query Assistant

*Ask questions and get comprehensive answers about your research papers*

### Video Demo

[
*Click to watch a demonstration of ResearchWebGraph in action*

## Installation

### Prerequisites

1. Python 3.9 or higher
2. Docker (for running Qdrant locally)
3. Node.js (optional, if building additional frontend components)

### Backend Setup

1. Clone the repository:

```
git clone https://github.com/tirth8205/ResearchWebGraph.git
cd ResearchWebGraph/backend
```

2. Create a virtual environment:

```
python -m venv researchwebgraph_env
source researchwebgraph_env/bin/activate  # On Windows, use researchwebgraph_env\Scripts\activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Start Qdrant (if not already running):

```
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

5. Set up environment variables in `backend/.env`:

```
GROQ_API_KEY=your-groq-api-key
QDRANT_URL=http://localhost:6333
DEFAULT_LLM_MODEL=llama-3.1-8b-instant
```

6. Run the backend server:

```
uvicorn app.main:app --reload --port 8000
```


### Frontend Setup

1. Navigate to the frontend directory:

```
cd ../frontend
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Set up environment variables in `frontend/.env`:

```
BACKEND_URL=http://localhost:8000
```

4. Run the frontend application:

```
streamlit run app.py
```


## Usage

### Step 1: Search for Papers or Upload PDFs

1. Go to the "Research Papers" tab.
2. Search for papers on arXiv using keywords or upload your own PDF documents.

### Step 2: Build a Knowledge Graph

1. Select papers from the search results or uploaded PDFs.
2. Go to the "Knowledge Graph" tab and click "Build Knowledge Graph."
3. Explore the interactive graph to discover entities and relationships.

### Step 3: Ask Questions About the Papers

1. Go to the "Query Assistant" tab.
2. Ask specific questions about the selected papers (e.g., "What are the main findings?").
3. View detailed AI-generated answers with citations to source papers.

## Example Questions for Query Assistant

1. "What are the main findings across these papers?"
2. "Explain the methodology used in paper X."
3. "How do these papers relate to each other?"
4. "What are the limitations mentioned in these studies?"

## Project Structure

```
ResearchWebGraph/
├── backend/
│   ├── app/
│   │   ├── routers/          # API routes (papers, knowledge graph, query)
│   │   ├── services/         # Core logic (fetching, processing, graph building)
│   │   ├── utils/            # Utility functions (PDF processing, NLP)
│   │   ├── models/           # Pydantic schemas for request/response validation
│   │   └── main.py           # FastAPI entry point
│   ├── requirements.txt      # Backend dependencies
│   └── .env                  # Backend environment variables
├── frontend/
│   ├── app.py                # Streamlit entry point for frontend UI
│   ├── pages/                # Multi-page Streamlit app (Papers, Graphs, Assistant)
│   ├── components/           # Reusable UI components (API client, sidebar)
│   ├── requirements.txt      # Frontend dependencies
│   └── .env                  # Frontend environment variables
└── README.md                 # Project documentation
```


## Deployment

### Deploying Backend with Docker Compose (Optional)

Create a `docker-compose.yml` file:

```
version: '3'
services:
  qdrant:
    image: qdrant/qdrant:v1.x.x  # Replace with latest version tag if needed
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_storage:/qdrant/storage

  backend:
    build:
      context: ./backend/
    ports:
      - "8000:8000"
    env_file:
      - ./backend/.env

```

Run the deployment:

```
docker-compose up --build -d
```


### Deploying Frontend on Streamlit Cloud or Heroku (Optional)

You can deploy the Streamlit frontend on platforms like Streamlit Cloud or Heroku by configuring your `frontend/.env` file with the production backend URL.

## Contributing

Contributions are welcome! Here's how you can contribute:

1. **Bug Reports**: Open an issue describing the bug and steps to reproduce it
2. **Feature Requests**: Open an issue describing the feature and its potential implementation
3. **Code Contributions**: Fork the repository, make your changes, and submit a pull request

### Development Guidelines

- Follow the existing code style and patterns
- Write tests for new features
- Update documentation for any changes


## Feedback \& Community

We value your feedback! Here's how to reach us:

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Join our [GitHub Discussions](https://github.com/tirth8205/ResearchWebGraph/discussions) for questions and ideas
- **Contact**:
    - Name: Tirth Kanani
    - Email: tirthkanani18@gmail.com
    - LinkedIn: [https://www.linkedin.com/in/tirthkanani/](https://www.linkedin.com/in/tirthkanani/)

Your feedback helps make ResearchWebGraph better for everyone!

## License

This project is licensed under the MIT License.