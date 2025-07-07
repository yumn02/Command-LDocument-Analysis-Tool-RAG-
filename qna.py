import argparse
import os
import fitz
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from chromadb.config import Settings

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
chroma_client = chromadb.PersistentClient(path="db")
collection = chroma_client.get_or_create_collection(name="documents")

def load_document(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text = ""
        doc = fitz.open(path)
        for page in doc:
            text += page.get_text()
        return text
    elif ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError("Only PDF and TXT files are supported.")

def split_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def get_embedding(text):
    return genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )["embedding"]

parser = argparse.ArgumentParser(description="Document Q&A Tool")
subparsers = parser.add_subparsers(dest="command")

index_parser = subparsers.add_parser("index", help="Index a document")
index_parser.add_argument("--path", required=True, help="Path to the document (PDF or TXT)")

ask_parser = subparsers.add_parser("ask", help="Ask a question")
ask_parser.add_argument("--question", required=True, help="Your question")

args = parser.parse_args()

if args.command == "index":
    try:
        text = load_document(args.path)
        chunks = split_text(text)
        embeddings = []
        final_chunks = []
        final_ids = []

        for i, chunk in enumerate(chunks):
            try:
                emb = get_embedding(chunk)
                embeddings.append(emb)
                final_chunks.append(chunk)
                final_ids.append(f"chunk-{i}")
            except Exception as e:
                print(f"Couldn't embed part {i}: {e}")

        if embeddings:
            collection.delete(where={"source": "indexed"})
            collection.add(
                documents=final_chunks,
                embeddings=embeddings,
                ids=final_ids,
                metadatas=[{"source": "indexed"} for _ in final_chunks]
            )
            print("Document saved successfully.")
        else:
            print("No parts were saved.")

    except Exception as e:
        print(f"Something went wrong: {e}")

elif args.command == "ask":
    try:
        question_embedding = get_embedding(args.question)
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=5,
            where={"source": "indexed"}
        )
        if not results["documents"] or not results["documents"][0]:
            print("No answer found.")
            exit()

        relevant_chunks = []
        for doc_list in results["documents"]:
            relevant_chunks.extend(doc_list)
        context = "\n".join(relevant_chunks)

        prompt = f"""
Use the following document to answer the question.

Document:
{context}

Question:
{args.question}

Answer in a clear and simple way.
"""

        chat = genai.GenerativeModel("gemini-1.5-flash").start_chat()
        response = chat.send_message(prompt)

        print("Answer:")
        print(response.text.strip())

    except Exception as e:
        print(f"Something went wrong: {e}")
