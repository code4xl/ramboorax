# LLM-Powered Query Retrieval System 🚀

An intelligent document processing and query-retrieval system that uses AI to analyze insurance policies, legal documents, and compliance materials, providing accurate answers with contextual understanding.

## ✨ Features

- **Smart Document Processing**: Automatically extracts and chunks text from PDF and DOCX files
- **Semantic Search**: Uses OpenAI embeddings with FAISS for intelligent content retrieval
- **AI-Powered Answers**: GPT-4 integration for contextual query understanding and response generation
- **RESTful API**: FastAPI-based backend with comprehensive documentation
- **Explainable Results**: Provides reasoning and source references for all answers

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.8+
- **AI/ML**: OpenAI GPT-4, OpenAI Embeddings, FAISS
- **Document Processing**: PyPDF, python-docx
- **Database**: In-memory storage (easily extensible to PostgreSQL/Pinecone)

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd server
   ```

2. **Install dependencies**
   ```bash
   pip install fastapi uvicorn pydantic httpx python-dotenv openai pypdf python-docx faiss-cpu numpy networkx python-multipart
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

4. **Run the server**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## 📖 API Usage

### Main Endpoint

**POST** `/hackrx/run`

Process documents and answer queries:

```json
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "Does this policy cover maternity expenses?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "A grace period of thirty days is provided for premium payment...",
    "Yes, the policy covers maternity expenses with specific conditions..."
  ]
}
```

### Health Check

**GET** `/hackrx/health`

Check service status.

## 📚 API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🏗️ Project Structure

```
server/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── models/
│   │   └── hackrx.py          # Request/Response models
│   ├── routes/
│   │   └── query_retrieval.py # API endpoints
│   └── services/
│       ├── document_processor.py # PDF/DOCX processing
│       ├── embedding_service.py  # Vector embeddings
│       └── query_processor.py    # Query handling
├── requirements.txt
└── .env
```

## 🧪 Testing

Test the installation:

```python
# test_imports.py
from pypdf import PdfReader
import openai
import faiss
print("✅ All dependencies installed successfully!")
```

## 📈 Performance

- **Accuracy**: High precision query understanding and clause matching
- **Speed**: Optimized embedding search with FAISS
- **Scalability**: Modular architecture for easy scaling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the logs for detailed error information

---

**Built with ❤️ for intelligent document processing**
