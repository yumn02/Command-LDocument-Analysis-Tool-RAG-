# Task 3 - Document Q&A Tool (RAG)

This is a small Python project I built for Task 3.

I made a command-line tool that lets you ask questions about a document (PDF or TXT).  
First, the tool reads the file and splits it into smaller parts. Then it uses Gemini to create embeddings for each part.  
The embeddings are saved locally using ChromaDB.

After that, you can ask any question about the document.  
The tool finds the most related parts and sends them (with the question) to Gemini to get a final answer.  
The answer is printed directly in the terminal.

## How to use

### 1. Index a document

This command reads the document and saves the chunks:
python qna.py index --path yourfile.pdf

### 2. Ask a question

This command takes your question and gives an answer based on the document:
python qna.py ask --question "What is this document about?"

## Notes

- The tool supports `.pdf` and `.txt` files.
- All answers appear in the terminal only.
- You need to add your Gemini API key in the `.env`:
  GEMINI_API_KEY=add_your_api_key_here

That's it. I kept it simple and clear so it's easy to use and understand.
