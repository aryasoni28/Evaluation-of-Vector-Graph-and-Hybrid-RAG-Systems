# Evaluation-of-Vector-Graph-and-Hybrid-RAG-Systems
#  RAG Performance Evaluation System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![Neo4j](https://img.shields.io/badge/Graph-Neo4j-008CC1)
![ChromaDB](https://img.shields.io/badge/Vector-ChromaDB-32CD32)
![Gemini](https://img.shields.io/badge/LLM-Gemini-FF6F00)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive evaluation platform for comparing Retrieval-Augmented Generation (RAG) architectures with detailed performance metrics and visualization capabilities.

##  Features

###  Multi-Architecture Support
- **Naive RAG** - Traditional vector retrieval with ChromaDB
- **Graph RAG** - Knowledge graph-powered retrieval with Neo4j
- **Hybrid RAG** - Combined vector + graph approach

###  Evaluation Metrics
| Category | Metrics |
|----------|---------|
| **Retrieval** | Precision@k, Recall@k, Context Relevance |
| **Generation** | BLEU Score, F1 Score, Answer Quality |
| **Performance** | Latency, Memory Usage, Throughput |
| **Reliability** | Hallucination Detection, Error Rates |

###  Developer Tools
- Vector inspection and visualization
- Context debugging tools
- Memory and latency profiling
- Backup storage systems

##  Installation

### Prerequisites
- Python 3.8+
- Neo4j Desktop/Server (for Graph RAG)
- Google Gemini API key

### Setup Steps
```bash
# Clone repository
git clone https://github.com/yourusername/rag-evaluation.git
cd rag-evaluation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```
Usage
streamlit run app.py
Workflow
Upload Documents (PDF/TXT)

Enter Query and optional ground truth

View Results from all three RAG systems

Analyze Metrics through interactive visualizations
![image](https://github.com/user-attachments/assets/7a7fe654-d907-4c15-8dab-08a8230f4880)
 Configuration Options
Environment Variable	Description	Default
GEMINI_API_KEY	Google Gemini API key	Required
NEO4J_URI	Neo4j connection URI	bolt://localhost:7687
CHROMA_DB_PATH	ChromaDB storage path	./chroma_db
VECTOR_CACHE_SIZE	Embedding cache size (MB)	500
