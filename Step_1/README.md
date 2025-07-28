# RAG Meta Financial QA - Step 1

A Retrieval-Augmented Generation (RAG) system for answering questions about Meta's Q1 2024 Financial Report using document processing, vector embeddings, and AI-powered response generation.

## 📋 Overview

This system processes Meta's Q1 2024 Financial Report PDF, creates searchable text chunks with vector embeddings, stores them in a LanceDB vector database, and uses Google's Gemini AI to answer financial questions based on the retrieved context.

## 🚀 Features

- **PDF Document Processing**: Extracts and processes text from Meta's financial report
- **Intelligent Chunking**: Breaks documents into meaningful chunks with overlap
- **Vector Embeddings**: Uses Sentence Transformers for semantic search capabilities
- **Vector Database**: LanceDB for efficient similarity search
- **AI Response Generation**: Google Gemini for natural language responses
- **Evaluation Framework**: Built-in testing with expected answers

## 📋 Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- Google API Key for Gemini AI

## 🛠 Installation

1. **Clone the repository and navigate to Step 1:**

   ```bash
   cd RAG_Meta_Financial_QA/Step_1
   ```

2. **Create and activate virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file with your Google API key:

   ```bash
   GOOGLE_API_KEY=your_google_api_key_here
   LANCEDB_PATH=data/sample-lancedb
   ```

5. **Ensure the PDF file exists:**
   Make sure `data/source/Meta's Q1 2024 Financial Report.pdf` is in place.

## 🎯 Quick Start

### Option 1: Run the Complete Pipeline (Recommended)

```bash
python run_step1.py
```

This will:

- Initialize the RAG pipeline
- Reset the database
- Process the Meta financial report
- Run test queries
- Save results to `outputs/step1_outputs.json`

### Option 2: Use the Command-Line Interface

**Add documents to the database:**

```bash
python main.py add -p "data/source/Meta's Q1 2024 Financial Report.pdf"
```

**Run test queries:**

```bash
python main.py test_step1
```

**Query the system:**

```bash
python main.py query "What was Meta's revenue in Q1 2024?"
```

**Reset the database:**

```bash
python main.py reset
```

**Run the full pipeline:**

```bash
python main.py run -p "data/source/Meta's Q1 2024 Financial Report.pdf"
```

## 📊 Expected Results

### Sample Output from `run_step1.py`:

```
Initializing pipeline...
🗑️ Resetting the database...
🔍 Adding document: data/source/Meta's Q1 2024 Financial Report.pdf
📝 Adding documents...
✅ Added 41 items to the datastore

🔍 Checking datastore content...
Datastore contains 41 items

❓ Test Query: What was Meta's revenue in Q1 2024?
🤖 Response: Meta's total revenue in Q1 2024 was $36,455 million.
✅ Expected: $36.455 billion

❓ Test Query: What were the key financial highlights for Meta in Q1 2024?
🤖 Response: [Financial highlights response]
✅ Expected: Revenue: $36.455 billion (27% increase), Net income: $12.369 billion (117% increase), Operating margin: 38%, EPS: $4.71, DAP: 3.24 billion (7% increase)

✅ Outputs saved to outputs/step1_outputs.json
```

## 📁 Project Structure

```
Step_1/
├── main.py                    # CLI interface
├── run_step1.py              # Direct pipeline runner
├── create_parser.py          # Command-line argument parser
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables
├── data/
│   ├── source/              # Input documents
│   └── sample-lancedb/      # Vector database
├── outputs/
│   └── step1_outputs.json   # Test results
└── src/
    ├── rag_pipeline.py      # Main pipeline orchestrator
    ├── interface/           # Abstract base classes
    ├── impl/               # Implementation classes
    └── util/               # Utility functions
```

## 🔧 Configuration

### Environment Variables (.env)

```bash
GOOGLE_API_KEY=your_google_api_key_here
LANCEDB_PATH=data/sample-lancedb
```

### Key Parameters

- **Chunk Size**: 400 tokens (configurable in `indexer.py`)
- **Chunk Overlap**: 50 tokens
- **Embedding Model**: `all-MiniLM-L6-v2`
- **Vector Dimensions**: 384
- **Search Results**: Top 5 most similar chunks

## 🧪 Testing

The system includes built-in test queries to validate functionality:

1. **Revenue Query**: "What was Meta's revenue in Q1 2024?"

   - Expected: $36.455 billion

2. **Financial Highlights**: "What were the key financial highlights for Meta in Q1 2024?"
   - Expected: Multiple financial metrics including revenue growth, net income, etc.

Results are saved to `outputs/step1_outputs.json` for analysis.

## 🔍 Troubleshooting

### Common Issues:

1. **File Not Found Error:**

   ```bash
   ❌ File not found: data/source/Meta's Q1 2024 Financial Report.pdf
   ```

   **Solution**: Ensure the PDF file exists in the correct location.

2. **Import Errors:**

   ```bash
   ModuleNotFoundError: No module named 'sentence_transformers'
   ```

   **Solution**: Install dependencies: `pip install -r requirements.txt`

3. **API Key Error:**

   ```bash
   Error: Google API key not found
   ```

   **Solution**: Set up your `.env` file with a valid `GOOGLE_API_KEY`.

4. **Database Corruption:**
   ```bash
   lance error: Query Execution error
   ```
   **Solution**: The system automatically resets the database. If issues persist, manually delete `data/sample-lancedb/` directory.

### Debug Mode:

Run with verbose logging to see detailed processing steps:

```bash
python -c "import logging; logging.basicConfig(level=logging.DEBUG)" && python run_step1.py
```

## 📈 Performance

- **Document Processing**: ~30 seconds for 10-page PDF
- **Embedding Generation**: ~2-3 seconds for 41 chunks
- **Query Response**: ~2-5 seconds per query
- **Database Storage**: 41 text chunks from Meta's Q1 2024 report

## 🔄 Pipeline Components

1. **Indexer**: Processes PDFs and creates text chunks
2. **Datastore**: Manages vector embeddings and similarity search
3. **Retriever**: Finds relevant context for queries
4. **Response Generator**: Uses Gemini AI for natural language responses
5. **Evaluator**: Tests system performance against expected answers

## 📝 Output Format

Results are saved in JSON format:

```json
[
  {
    "query": "What was Meta's revenue in Q1 2024?",
    "response": "Meta's total revenue in Q1 2024 was $36,455 million.",
    "expected_answer": "$36.455 billion"
  }
]
```

## 🛡️ Requirements

See `requirements.txt` for complete list. Key dependencies:

- `docling`: PDF processing
- `sentence-transformers`: Text embeddings
- `lancedb`: Vector database
- `google-generativeai`: AI response generation

