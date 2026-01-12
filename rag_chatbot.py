import os
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 1. Cargar PDF
loader = PyPDFLoader("documento.pdf")
documents = loader.load()

# 2. Dividir texto en fragmentos
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# 3. Crear embeddings y base vectorial
embeddings = OpenAIEmbeddings(api_key=api_key)
db = FAISS.from_documents(texts, embeddings)

# 4. Pregunta del usuario
query = "¿De qué trata el documento?"

docs = db.similarity_search(query)

# 5. Enviar contexto recuperado al modelo
client = OpenAI(api_key=api_key)

context = "\n".join([doc.page_content for doc in docs])

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": "Responde solo usando la información del contexto."},
        {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {query}"}
    ]
)

print(response.choices[0].message.content)
