import streamlit as st
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPEN_API_KEY")
st.title("Question Answering Chatbot")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    pdf_reader = PdfReader(uploaded_file)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    st.write("### Extracted PDF Text:")
    st.write(pdf_text[:1000])  
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text_chunks = text_splitter.split_text(pdf_text)
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.5)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    user_question = st.text_input("Ask a question based on the PDF:")

    if user_question:
        relevant_docs = vector_store.similarity_search(user_question, k=3)
        qa_chain = load_qa_chain(llm)
        answer = qa_chain.run(input_documents=relevant_docs, question=user_question)
        st.write("### Answer:")
        st.write(answer)
