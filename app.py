import streamlit as st
import pickle
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os

load_dotenv()

def add_vertical_space(lines: int):
    """Adds vertical space in the Streamlit app."""
    for _ in range(lines):
        st.write("")

with st.sidebar:
    st.title("LLM CHAT APP SAMPLE")
    st.markdown('''
    ## About
    This app is a LLM powered chatbot built using:
    
    - [Streamlit](https://streamlit.io/)
    - [Langchain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM Model
    ''')  
    add_vertical_space(5)
    st.write('Made by Anushka Agarwal')

def main():
    st.header("Chat using PDF :)")
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        st.write(f"File name: {pdf.name}")
        st.success("PDF loaded successfully!")

        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        st.subheader("PDF Text Chunks:")
        st.write(chunks)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

        store_name = pdf.name[:-4]  
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(vectorstore, f)
            st.success(f"Vector store saved as {store_name}.pkl")
        query = st.text_input("Ask Questions about your PDF file: ") 
        #st.write(query)
        
        if query:
            docs = vectorstore.similarity_search(query=query,k=3)
            #st.write(docs)
            llm= OpenAI(temperature=0,)
            chain = load_qa_chain(llm=llm,chain_type="stuff")
            response = chain.run(input_documents = docs , question= query)
            st.write(response)

    else:   
        st.info("Please upload a PDF to begin.")

    add_vertical_space(2)

if __name__ == "__main__":
    main()
