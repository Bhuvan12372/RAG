'''import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
hf_token = os.getenv('HF_TOKEN')
llm = ChatGroq(model_name = 'llama3-8b-8192',groq_api_key = groq_api_key)
prompt = ChatPromptTemplate.from_template(
    """
    Answers the question based on the provided context only.
    please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question : {input}

"""
)

def create_vector_embeddings():
    # i will use session_state to store this memory to use this code when i needs in some other functions 
    # because vector embeddings is a computationally expensive.so we store this process in session state 
    # and uses whenever we needs.
    if "vectorstorage" not in st.session_state:  #creating the vectors variable
        st.session_state.loader = PyPDFDirectoryLoader("papers")
        st.session_state.load = st.session_state.loader.load()
        st.session_state.Recursively = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 100)
        st.session_state.final_docs = st.session_state.Recursively.split_documents(st.session_state.load[:2]) # i am taking only top 2 documents to avoid time complexity
        st.session_state.embeddings = HuggingFaceBgeEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2",hf_token = hf_token)
        st.session_state.vectorstorage = FAISS.from_documents(
                st.session_state.final_docs, 
                st.session_state.embeddings
            )

st.title('RAG Document')
user_prompt = st.text_input("Enter the questions from the document")
if st.button("Vector_embedding"):
    create_vector_embeddings()
    st.success("Vector embeddings created successfully!")

import time

if user_prompt:
    document_chain = create_stuff_documents_chain(prompt=prompt,llm=llm)
    # what does this function does is if we have more then 1 no.of 
    # pdf's then above function connects or combines the pdf's
    retreiver = st.session_state.vectorstorage.as_retriever()
    # retreiver used to retreive the required amount of answer from the corpus as it overcomes the limtations from embedding techniques
    retreiver_chain = create_retrieval_chain(retreiver,document_chain)
    start = time.process_time()
    response = retreiver_chain.invoke({'input':user_prompt})
    print(f"time taken to complete the process is {time.process_time()-start}")
    st.write(response['answer'])
    with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('--------------------------------')'''
        

import os
import time
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
CHUNK_SIZE = 700
CHUNK_OVERLAP = 30
PDF_DIRECTORY = "papers"
MODEL_NAME = "llama3-8b-8192"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Initialize environment variables
def init_environment():
    groq_api_key = os.getenv('GROQ_API_KEY')
    hf_token = os.getenv('HF_TOKEN')
    
    if not groq_api_key or not hf_token:
        st.error("Missing required environment variables. Please check .env file.")
        st.stop()
    
    # Set HuggingFace token in environment
    os.environ['HUGGINGFACE_API_TOKEN'] = hf_token
    
    return groq_api_key

# Initialize LLM and prompt template
def init_llm_and_prompt(groq_api_key):
    llm = ChatGroq(
        model_name=MODEL_NAME,
        groq_api_key=groq_api_key
    )
    
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    
    Context: {context}
    
    Question: {input}
    """)
    
    return llm, prompt

# Create vector embeddings
def create_vector_embeddings():
    """Create vector embeddings if not already in session state"""
    if "vectorstorage" not in st.session_state:
        with st.spinner("Creating vector embeddings..."):
            try:
                
                loader = PyPDFDirectoryLoader(PDF_DIRECTORY)
                documents = loader.load()
            
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP
                )
                split_docs = splitter.split_documents(documents[:2])  
                
                embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                vectorstore = FAISS.from_documents(split_docs, embeddings)
                
                # Store in session state
                st.session_state.vectorstorage = vectorstore
                return True
                
            except Exception as e:
                st.error(f"Error creating embeddings: {str(e)}")
                return False
    return True

def process_query(user_prompt, llm, prompt):
    try:
        # Create document chain
        document_chain = create_stuff_documents_chain(
            prompt=prompt,
            llm=llm
        )
        
        # Create retriever
        retriever = st.session_state.vectorstorage.as_retriever()
        retriever_chain = create_retrieval_chain(retriever, document_chain)
        
        # Measure processing time
        start_time = time.process_time()
        response = retriever_chain.invoke({'input': user_prompt})
        processing_time = time.process_time() - start_time
        
        return response, processing_time
        
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None, None

def main():
    st.title('RAG Document Query System')
    st.write("Upload PDFs and ask questions about their content.")
    
    # Initialize environment
    groq_api_key = init_environment()
    llm, prompt = init_llm_and_prompt(groq_api_key)
    
    # User input
    user_prompt = st.text_input("Enter your question about the documents:", 
                               placeholder="What would you like to know?")
    
    # Create embeddings button
    if st.button("Initialize Vector Embeddings"):
        if create_vector_embeddings():
            st.success(" Vector embeddings created successfully!")
    
    # Process query
    if user_prompt and "vectorstorage" in st.session_state:
        response, processing_time = process_query(user_prompt, llm, prompt)
        
        if response:
            st.write("### Answer")
            st.write(response['answer'])
            st.info(f"Processing time: {processing_time:.2f} seconds")
            
            with st.expander("Document Similarity Search Results"):
                for i, doc in enumerate(response['context'], 1):
                    st.markdown(f"**Relevant Extract {i}:**")
                    st.write(doc.page_content)
                    st.divider()
    elif user_prompt:
        st.warning("Please initialize vector embeddings first!")

if __name__ == "__main__":
    main()