import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
import sqlite3
import time

# Enable caching for LangChain
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# Configure Streamlit for better performance
st.set_page_config(
    page_title="Fast LangChain Demo",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def initialize_environment():
    """Initialize environment variables with caching"""
    load_dotenv()
    return {
        'langchain_api_key': os.getenv('LANGCHAIN_API_KEY'),
        'langchain_project': os.getenv('LANGCHAIN_PROJECT'),
        'groq_api_key': os.getenv('GROQ_API_KEY')
    }

# Initialize environment
env_vars = initialize_environment()

# Set environment variables
os.environ['LANGCHAIN_API_KEY'] = env_vars['langchain_api_key']
os.environ['LANGCHAIN_PROJECT'] = env_vars['langchain_project']
os.environ['LANGCHAIN_TRACING_V2'] = "true"
groq_api_key = env_vars['groq_api_key']

@st.cache_resource
def create_model():
    """Create and cache the model instance"""
    return ChatGroq(
        model='llama3-8b-8192',
        groq_api_key=groq_api_key,
        temperature=0.7,  
        max_tokens=2048,  
        streaming=True    
    )

@st.cache_resource
def create_prompt_template():
    """Create and cache the prompt template"""
    return ChatPromptTemplate.from_messages([
        ("system", """You are a highly efficient AI assistant optimized for fast, accurate responses. 
        Keep answers concise yet informative. Focus on key points and deliver them clearly.
        If you don't know something, say so immediately rather than elaborating."""),
        ("user", "Question: {question}")
    ])

# Initialize model and prompt template
model = create_model()
prompt = create_prompt_template()
output_parser = StrOutputParser()

# Create the chain with caching
@st.cache_resource
def create_chain():
    """Create and cache the processing chain"""
    return prompt | model | output_parser

chain = create_chain()

# UI Components
st.title(" Fast LangChain Demo with Llama3")

# Add a session state for tracking processing time
if 'processing_times' not in st.session_state:
    st.session_state.processing_times = []

# Create sidebar for metrics
with st.sidebar:
    st.title("Performance Metrics")
    if st.session_state.processing_times:
        avg_time = sum(st.session_state.processing_times) / len(st.session_state.processing_times)
        st.metric("Average Response Time", f"{avg_time:.2f}s")
        st.metric("Fastest Response", f"{min(st.session_state.processing_times):.2f}s")
        st.metric("Total Queries", len(st.session_state.processing_times))

@st.cache_data(ttl=3600)  
def get_response(question: str):
    """Get cached response for a question"""
    return chain.invoke({"question": question})

user_input = st.text_input(
    "What question do you have in your mind?",
    key="question_input",
    help="Type your question here and press Enter"
)

if user_input:
    try:
        # Add a loading spinner
        with st.spinner("Processing your question..."):
            # Track processing time
            start_time = time.time()
            
            # Get response with caching
            response = get_response(user_input)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            st.session_state.processing_times.append(processing_time)
            
            # Display response in a clean container
            with st.container():
                st.write("Response:")
                st.write(response)
                st.caption(f"Processed in {processing_time:.2f} seconds")
                
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        st.error("Please try again or contact support if the error persists.")

# Add clear button
if st.button("Clear Cache"):
    try:
        # Clear Streamlit cache
        st.cache_data.clear()
        # Clear SQLite cache
        conn = sqlite3.connect(".langchain.db")
        c = conn.cursor()
        c.execute("DELETE FROM langchain_cache")
        conn.commit()
        conn.close()
        # Clear session state
        st.session_state.processing_times = []
        st.success("Cache cleared successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"Error clearing cache: {str(e)}")