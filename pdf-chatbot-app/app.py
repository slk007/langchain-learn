import os
import pickle

import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from streamlit_extras.add_vertical_space import add_vertical_space

with st.sidebar:
    st.title("LLM Chat App")
    st.markdown("""App uses Streamlit, Langchain, OpenAI """)
    add_vertical_space(5)
    st.write("Made using Youtube tutorial")

load_dotenv()


def main():
    st.header("Chat with PDF")

    # upload PDF
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf:
        # read PDF
        pdf_reader = PdfReader(pdf)
        # st.write(pdf.name)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text=text)

        pdf_name = pdf.name[:-4]
        file_name = f"pickles/{pdf_name}.pkl"

        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                vector_store = pickle.load(f)
        # Embeddings loaded from the disk
        else:
            # embeddings
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            with open(file_name, "wb") as f:
                pickle.dump(vector_store, f)
        # Embeddings Computation Completed

        # Accept user questions query
        query = st.text_input("Ask questions about your PDF")
        if query:
            docs = vector_store.similarity_search(query=query, k=3)
            llm = OpenAI(temperature=0)
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)


if __name__ == "__main__":
    main()
