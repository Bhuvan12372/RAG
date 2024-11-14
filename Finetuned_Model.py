import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
from sklearn.metrics import accuracy_score
import sqlite3
import time
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
from peft import LoraConfig, get_peft_model
import json
import logging
from typing import Optional, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable caching for LangChain
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# Configure Streamlit
st.set_page_config(
    page_title="Finetuned LangChain Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_environment() -> Dict[str, Optional[str]]:
    """Initialize environment variables with caching"""
    load_dotenv()
    return {
        'langchain_api_key': os.getenv('LANGCHAIN_API_KEY'),
        'langchain_project': os.getenv('LANGCHAIN_PROJECT'),
        'groq_api_key': os.getenv('GROQ_API_KEY'),
        'huggingface_token': os.getenv('HUGGINGFACE_TOKEN')
    }

from transformers import TrainerCallback

class ProgressCallback(TrainerCallback):
    def __init__(self, progress_bar):
        self.progress_bar = progress_bar
        self.total_steps = 0
        self.current_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training"""
        self.total_steps = state.max_steps
        self.current_step = 0
        if self.progress_bar is not None:
            self.progress_bar.progress(0)

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each step"""
        self.current_step += 1
        if self.progress_bar is not None:
            current_progress = min(1.0, self.current_step / self.total_steps)
            self.progress_bar.progress(current_progress)
from sklearn.model_selection import train_test_split

class ModelFinetuner:
    def __init__(self, base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v0.1", hf_token: Optional[str] = None):
        self.base_model_name = base_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        try:
            model_args = {
                "trust_remote_code": True,
                "device_map": "auto" if torch.cuda.is_available() else None,
                **({"token": hf_token} if hf_token else {})
            }
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                **model_args
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load model with proper configuration for CPU/GPU
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                **model_args
            )
            
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
                
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def prepare_dataset(self, data_path: str) -> tuple[Dataset, Dataset]:
        """Prepare train and eval datasets for finetuning with improved error handling"""
        try:
            # Load and validate data
            if data_path.endswith('.json'):
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif data_path.endswith('.csv'):
                data = pd.read_csv(data_path).to_dict('records')
            else:
                raise ValueError("Unsupported file format. Use JSON or CSV.")
            
            # Validate data structure
            if not all('question' in item and 'answer' in item for item in data):
                raise ValueError("Data must contain 'question' and 'answer' fields")
            
            # Format the text for training
            formatted_texts = []
            for item in data:
                text = f"### Question: {item['question']}\n### Answer: {item['answer']}"
                formatted_texts.append(text)
            
            # Create dataset
            dataset_dict = {'text': formatted_texts}
            full_dataset = Dataset.from_dict(dataset_dict)
            
            # Split into train and eval datasets
            train_test_dict = full_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
            train_data = train_test_dict['train']
            eval_data = train_test_dict['test']

            def tokenize_function(examples):
                # Tokenize the texts with padding and truncation
                outputs = self.tokenizer(
                    examples['text'],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors=None  # Return as lists, not tensors
                )
                
                # Prepare the labels (same as input_ids for causal language modeling)
                outputs["labels"] = outputs["input_ids"].copy()
                
                return outputs
            
            # Apply tokenization to both datasets
            tokenized_train = train_data.map(
                tokenize_function,
                batched=True,
                remove_columns=train_data.column_names
            )
            
            tokenized_eval = eval_data.map(
                tokenize_function,
                batched=True,
                remove_columns=eval_data.column_names
            )
            
            return tokenized_train, tokenized_eval
            
        except Exception as e:
            logger.error(f"Dataset preparation failed: {str(e)}")
            raise

    def setup_training(self, learning_rate: float = 2e-5, num_epochs: int = 3) -> TrainingArguments:
        """Setup LoRA training configuration with improved memory management"""
        try:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Apply LoRA config
            self.model = get_peft_model(self.model, lora_config)
            
            # Training arguments with memory optimizations
            training_args = TrainingArguments(
                output_dir="./finetuned_model",
                learning_rate=learning_rate,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=1,  # Reduced batch size
                per_device_eval_batch_size=1,   # Reduced batch size
                gradient_accumulation_steps=4,
                fp16=torch.cuda.is_available(),
                logging_dir="./logs",
                logging_steps=10,
                evaluation_strategy="steps",
                eval_steps=50,
                save_strategy="steps",
                save_steps=50,
                save_total_limit=2,
                load_best_model_at_end=True,
                remove_unused_columns=False,  # Important for custom datasets
            )
            
            return training_args
            
        except Exception as e:
            logger.error(f"Training setup failed: {str(e)}")
            raise

    def train(self, train_dataset: Dataset, eval_dataset: Dataset, training_args: TrainingArguments, progress_bar=None) -> None:
        """Train the model with performance metrics logging"""
        try:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                callbacks=[ProgressCallback(progress_bar)] if progress_bar is not None else []
            )
            
            # Start training
            trainer.train()

            # After training, evaluate the model
            logger.info("Evaluating model on the validation set...")
            eval_results = trainer.evaluate()

            # Log and display performance metrics
            st.write("### Evaluation Results")
            for key, value in eval_results.items():
                st.write(f"{key}: {value}")

            # Save model and tokenizer
            output_dir = "./finetuned_model"
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise


def create_chat_model(groq_api_key: str) -> Optional[ChatGroq]:
    """Create ChatGroq model instance with error handling"""
    try:
        return ChatGroq(
            model_name='mixtral-8x7b-32768',
            groq_api_key=groq_api_key,
            temperature=0.7,
            max_tokens=2048,
            streaming=True
        )
    except Exception as e:
        logger.error(f"Failed to create ChatGroq model: {str(e)}")
        return None

# Initialize environment
env_vars = initialize_environment()

# Set environment variables
for key, value in env_vars.items():
    if value:
        os.environ[key.upper()] = value

# Streamlit UI
st.title("🤖 LangChain Demo with Model Finetuning")

# Sidebar for finetuning
with st.sidebar:
    st.title("Model Finetuning")
    
    uploaded_file = st.file_uploader(
        "Upload training data (JSON/CSV)", 
        type=['json', 'csv']
    )
    
    learning_rate = st.slider(
        "Learning Rate", 
        min_value=1e-6, 
        max_value=1e-4, 
        value=2e-5, 
        format="%.6f"
    )
    
    num_epochs = st.number_input(
        "Number of Epochs", 
        min_value=1, 
        max_value=10, 
        value=3
    )
    
    if uploaded_file and st.button("Start Finetuning"):
        try:
            with st.spinner("Preparing for finetuning..."):
                # Save uploaded file
                file_path = f"training_data.{uploaded_file.name.split('.')[-1]}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Initialize model and prepare datasets
                finetuner = ModelFinetuner(hf_token=env_vars['huggingface_token'])
                train_dataset, eval_dataset = finetuner.prepare_dataset(file_path)
                
                # Setup training arguments
                training_args = finetuner.setup_training(
                    learning_rate=learning_rate,
                    num_epochs=num_epochs
                )
                
                st.info("Starting training... This may take a while.")
                
                # Create progress bar in Streamlit
                progress_bar = st.progress(0)
                
                # Execute training with progress bar
                finetuner.train(
                    train_dataset, 
                    eval_dataset, 
                    training_args,
                    progress_bar=progress_bar
                )
                
                st.success("✅ Training completed and model saved!")
        
        except Exception as e:
            st.error(f"Error during finetuning: {str(e)}")
            logger.error(f"Detailed error during finetuning: {str(e)}", exc_info=True)
    # Add this to your main Streamlit UI code below the finetuning section

# Create two columns: one for finetuning and one for Q&A
# Add this to your main Streamlit UI code below the finetuning section

# Create two columns: one for finetuning and one for Q&A
col1, col2 = st.columns(2)

with col1:
    st.title("Model Finetuning")
    # Your existing finetuning code here...
    # (Keep all the existing sidebar and finetuning logic in this column)

with col2:
    st.title("Ask Questions")
    
    # Check if finetuned model exists
    if os.path.exists("./finetuned_model"):
        # Create text input for questions
        user_question = st.text_area("Enter your question:", height=100)
        
        if st.button("Ask"):
            if user_question:
                try:
                    with st.spinner("Generating response..."):
                        # Load finetuned model and tokenizer
                        finetuned_model = AutoModelForCausalLM.from_pretrained(
                            "./finetuned_model",
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None
                        )
                        finetuned_tokenizer = AutoTokenizer.from_pretrained("./finetuned_model")
                        
                        # Format the input
                        input_text = f"### Question: {user_question}\n### Answer:"
                        inputs = finetuned_tokenizer(
                            input_text, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True,
                            max_length=512
                        )
                        
                        # Move inputs to the same device as model if using GPU
                        if torch.cuda.is_available():
                            inputs = {k: v.cuda() for k, v in inputs.items()}
                        
                        # Generate response
                        outputs = finetuned_model.generate(
                            **inputs,
                            max_length=1024,
                            num_return_sequences=1,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=finetuned_tokenizer.pad_token_id
                        )
                        
                        # Decode and display response
                        response = finetuned_tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        # Extract just the answer part (after "### Answer:")
                        answer_part = response.split("### Answer:")[-1].strip()
                        
                        # Display in a nice format
                        st.markdown("### Response:")
                        st.write(answer_part)
                        
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    logger.error(f"Error in question answering: {str(e)}", exc_info=True)
            else:
                st.warning("Please enter a question first.")
    else:
        st.info("Please finetune the model first before asking questions.")
        
    # Add a section to show training data statistics if available
    if os.path.exists("training_data.json") or os.path.exists("training_data.csv"):
        st.markdown("### Training Data Statistics")
        try:
            if os.path.exists("training_data.json"):
                with open("training_data.json", 'r') as f:
                    train_data = json.load(f)
            else:
                train_data = pd.read_csv("training_data.csv").to_dict('records')
            
            st.write(f"Number of training examples: {len(train_data)}")
            
            # Show a few example Q&A pairs
            with st.expander("View example Q&A pairs"):
                for i, item in enumerate(train_data[:3]):  # Show first 3 examples
                    st.markdown(f"**Example {i+1}:**")
                    st.markdown(f"Q: {item['question']}")
                    st.markdown(f"A: {item['answer']}")
                    st.markdown("---")
                
        except Exception as e:
            st.error(f"Error loading training data statistics: {str(e)}")

