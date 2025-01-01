import os
import streamlit as st
import dill
import time
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain import FAISS
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env (especially OpenAI API key)

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    if data:
        # Split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)

        if docs:
            # Create embeddings and save to FAISS index
            embeddings = OpenAIEmbeddings()
            vectorstore_openai = FAISS.from_documents(docs, embeddings)
            main_placeholder.text("Embedding Vector Started Building...✅✅✅")
            time.sleep(2)

            # Temporarily remove the unpicklable components
            original_embedding_function = vectorstore_openai.embedding_function
            vectorstore_openai.embedding_function = None

            # Save the FAISS index to a pickle file
            with open(file_path, "wb") as f:
                dill.dump(vectorstore_openai, f)

            # Restore the removed components
            vectorstore_openai.embedding_function = original_embedding_function
        else:
            main_placeholder.text("Text Splitter produced empty documents. Check data.")
    else:
        main_placeholder.text("Data loading failed. Check URLs or network connection.")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = dill.load(f)

            # Restore the embedding function after loading
            vectorstore.embedding_function = OpenAIEmbeddings()

            # Create the QA chain
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            # Display the answer
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source + '\n')
