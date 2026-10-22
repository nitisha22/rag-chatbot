# University of Delhi RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about the University of Delhi's M.Sc. Informatics brochure using document retrieval and a hosted Large Language Model (LLM).

The application uses **Streamlit** for its web interface, **Sentence Transformers** to generate embeddings, **NumPy** to calculate cosine similarity, and **Groq** to generate answers from retrieved document context.

## Features

- Extract and chunk text from a PDF brochure.
- Generate text embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- Retrieve relevant document chunks using cosine similarity.
- Generate context-based answers using Groq's hosted LLM API.
- Display retrieved source chunks and PDF page references.
- Maintain conversation history during a Streamlit session.
- Keep API keys outside the source code using environment variables.

## Architecture

```text
University of Delhi Brochure (PDF)
              |
              v
      Text Extraction
              |
              v
        Text Chunking
              |
              v
     Generate Embeddings
              |
              v
      embeddings.json
              |
              v
       User Question
              |
              v
  Embed Question with Sentence
          Transformers
              |
              v
     Cosine Similarity Search
              |
              v
       Top 5 Text Chunks
              |
              v
   Question + Retrieved Context
              |
              v
        Groq LLM API
              |
              v
       Generated Answer
              |
              v
   Streamlit UI + Source Details
```

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| Anaconda | Python environment management |
| Sentence Transformers | Text embeddings |
| `all-MiniLM-L6-v2` | Embedding model |
| NumPy | Vector operations and cosine similarity |
| JSON | Storage of document chunks and embeddings |
| Groq API | Hosted LLM inference |
| `openai/gpt-oss-20b` | Answer generation |
| Streamlit | Web-based chatbot interface |
| python-dotenv | Loading API keys from `.env` |

## Project Structure

```text
RAG-Chatbot/
├── .env                    # Local API keys; do not commit
├── .gitignore
├── app.py                  # Streamlit web interface
├── rag_pipeline.py         # Retrieval and answer generation
├── test_groq.py            # Groq API connection test
├── test_gemini.py          # Earlier Gemini API test
├── week01/
│   ├── documents/           # Source PDF documents
│   ├── output/
│   │   ├── extracted_text.txt
│   │   └── embeddings.json
│   ├── embed_chunks.py      # Chunking and embedding generation
│   └── search_chunks.py     # Interactive retrieval test
└── README.md
```

*The structure above reflects the current development workflow; adjust it if your local project contains additional files or different filenames.*

## Prerequisites

- Python 3.12 or a compatible Python version
- Anaconda or Miniconda
- Git
- A Groq API key
- Internet access for API calls and initial model downloads

## Installation and Setup

### 1. Clone the repository

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RAG-Chatbot
```

Replace the placeholder with your repository URL.

### 2. Create and activate the Conda environment

If you are creating a fresh environment:

```powershell
conda create -n rag-chatbot python=3.12
conda activate rag-chatbot
```

If you already have the environment, activate it:

```powershell
conda activate rag-chatbot
```

### 3. Install dependencies

```powershell
python -m pip install numpy sentence-transformers groq streamlit python-dotenv
```

### 4. Configure the Groq API key

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_actual_groq_api_key
```

Replace the placeholder with your own API key from the [Groq Console](https://console.groq.com/keys).

**Security:** Never publish your API key, commit your `.env` file, or paste the key into public issues or screenshots. Keep `.env` excluded through `.gitignore`.

### 5. Prepare the document embeddings

The chatbot expects this file:

```text
week01/output/embeddings.json
```

It contains the document chunks, page numbers, and embedding vectors used for retrieval.

If the embeddings file is missing, place the source brochure in the expected input location and run your project's document-processing and embedding-generation scripts. For example, the embedding script is:

```powershell
python week01/embed_chunks.py
```

Follow the script's expected input paths and check that it creates `week01/output/embeddings.json` before starting the application.

### 6. Test the Groq connection (optional)

```powershell
python test_groq.py
```

A successful response confirms that your API key and configured model can generate text.

### 7. Start the chatbot

From the project root, run:

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

You can now ask questions about the M.Sc. Informatics brochure and expand the retrieved sources to inspect the supporting document chunks.

## Example Questions

Try asking:

1. What is the minimum percentage required for admission to M.Sc. Informatics?
2. What are the CUET-PG cutoff criteria and interview marks for admission?
3. What eligibility criteria are mentioned in the brochure?
4. What information does the brochure provide about the programme?

Answers depend on the content retrieved from the brochure. Always verify important admission requirements against the official university documents.

## How RAG Works in This Project

Retrieval-Augmented Generation combines document search with language generation.

1. **Embedding:** The user's question is converted into a vector using the same embedding model used for the document chunks.
2. **Retrieval:** The question vector is compared with stored document vectors using cosine similarity.
3. **Context construction:** The top five matching chunks are selected and combined with their source page numbers.
4. **Generation:** The question and retrieved context are sent to the configured Groq model.
5. **Presentation:** Streamlit displays the generated answer and the retrieved source chunks.

This helps ground responses in the supplied brochure rather than relying entirely on the model's pretrained knowledge. Retrieval errors and unsupported model-generated claims are still possible.

## Current Limitations

- Retrieval currently uses cosine similarity over locally stored JSON embeddings.
- Search quality depends on chunking and the wording of the user's question.
- The model can occasionally produce an incomplete or incorrect answer.
- The chatbot does not currently use a dedicated vector database.
- Source chunks show retrieved evidence, but generated page citations should still be verified.
- Availability, rate limits, and access to the configured Groq model depend on the provider's current terms and account limits.

## Future Improvements

- Improve chunking and retrieval accuracy.
- Add hybrid search or reranking.
- Evaluate answers against a set of representative questions.
- Improve source attribution and citation verification.
- Add conversation controls and a clearer error-handling experience.
- Deploy the application so others can access it online.

## Learning Goals

This project demonstrates practical implementation of:

- Document ingestion and text preprocessing
- Embedding generation and semantic search
- Retrieval-Augmented Generation
- Hosted LLM API integration
- Python application development
- Streamlit interface development
- Environment-variable and API-key management

## Disclaimer

This is an educational project based on a university brochure. It is not an official University of Delhi application. Refer to the university's official admissions materials for authoritative information.

## Author

Developed as a hands-on project to learn and implement a complete RAG chatbot workflow in Python.
