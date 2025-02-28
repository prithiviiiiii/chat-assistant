import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def extract_text_from_pdfs(pdf_files):
    """Extract text from a list of PDF files."""
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def split_text_into_chunks(text, chunk_size=10000, chunk_overlap=1000):
    """Split text into chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

def create_vector_store(text_chunks):
    """Create and save a FAISS vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def load_conversational_chain():
    """Load the conversational chain with a custom prompt template."""
    prompt_template = """
    Answer the question based on the provided context. Make sure descrribe as detail as possible and provide all tyhe details. If the answer is not available in the context, respond with "The answer is not available in the context."
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def handle_user_query(user_query):
    """Handle the user's query by searching the FAISS index and generating a response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    documents = vector_store.similarity_search(user_query)
    chain = load_conversational_chain()
    response = chain({"input_documents": documents, "question": user_query}, return_only_outputs=True)
    st.write("AI Response:", response["output_text"])

def main():
    st.set_page_config(page_title="Chat with PDF")
    st.header("Chat with PDF")

    user_query = st.text_input("Ask a question about the uploaded PDF files:")

    if user_query:
        handle_user_query(user_query)

    with st.sidebar:
        st.title("Upload and Process PDFs")
        pdf_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        if st.button("Process PDFs"):
            with st.spinner("Processing..."):
                raw_text = extract_text_from_pdfs(pdf_files)
                text_chunks = split_text_into_chunks(raw_text)
                create_vector_store(text_chunks)
                st.success("Processing complete!")




if __name__ == "__main__":
    main()