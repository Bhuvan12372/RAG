import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate  
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# langsmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
groq_api_key = os.getenv('GROQ_API_KEY')

# Setting prompt (How your AI should perform)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please answer the questions asked."),
        ("user", "Question: {question}")
    ]
)

# Streamlit framework
st.title("Langchain Demo with Llama3")

user_input = st.text_input("What question do you have in your mind?")

models = ChatGroq(model = 'llama3-8b-8192',groq_api_key= groq_api_key)

# Output parser
output_parser = StrOutputParser()

# Create the chain
chain = prompt | models | output_parser  # First data will go to prompt, then LLM, and finally output parser

# Handle the user input and show the response
if user_input:
    try:
        response = chain.invoke({"question": user_input})
        st.write(response)
    except Exception as e:
        st.error(f"Error occurred: {e}")


