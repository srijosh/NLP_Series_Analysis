import os
import re
import gc
import logging
import pandas as pd
import torch
import huggingface_hub
from datasets import Dataset
import transformers
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def remove_paranthesis(text):
    """Remove text within parentheses from the input string."""
    return re.sub(r'\(.*?\)', '', str(text)).strip()

class CharacterChatBot:
    def __init__(
        self,
        model_path,
        data_path="/content/NLP_Series_Analysis/data/naruto.csv",
        huggingface_token=None
    ):
        """
        Initialize the Character Chatbot with model and data configurations.
        
        Args:
            model_path (str): Path to save/load the model on Hugging Face Hub
            data_path (str): Path to the CSV file containing dialogue data
            huggingface_token (str, optional): Hugging Face authentication token
        """
        self.model_path = model_path
        self.data_path = data_path
        self.huggingface_token = huggingface_token
        self.base_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Login to Hugging Face if token provided
        if self.huggingface_token:
            try:
                huggingface_hub.login(self.huggingface_token)
            except Exception as e:
                logger.error(f"Hugging Face login failed: {e}")
                raise
        
        # Load or train model
        try:
            if huggingface_hub.repo_exists(self.model_path):
                logger.info(f"Loading existing model from {self.model_path}")
                self.model, self.model_tokenizer = self.load_model(self.model_path)
            else:
                logger.info("No existing model found. Training new model.")
                train_dataset = self.load_data()
                
                if len(train_dataset) == 0:
                    raise ValueError("No valid training data found")
                
                self.train(self.base_model_path, train_dataset)
                self.model, self.model_tokenizer = self.load_model(self.model_path)
        except Exception as e:
            logger.error(f"Model initialization error: {e}")
            raise

    def load_model(self, model_path):
        """
        Load a pre-trained model with quantization for efficient inference.
        
        Args:
            model_path (str): Path to the model
        
        Returns:
            transformers model and tokenizer
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        try:
            # Load model and tokenizer separately
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map=self.device
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Ensure pad token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return model, tokenizer
        
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            raise

    def chat(self, message, history):
        """
        Generate a response to a user message in the context of a conversation history.

        Args:
            message (str): Current user message
            history (list): Previous conversation messages

        Returns:
            str: Generated response in Naruto's character
        """
        system_prompt = """You are Naruto from the anime "Naruto". Your responses should reflect his personality and speech patterns. Be direct, enthusiastic, and to the point. Respond as if you're talking to a fellow ninja or friend."""

    # Prepare conversation history
        conversation_text = system_prompt + "\n"
    
        for user_msg, bot_resp in history:
            conversation_text += f"User: {user_msg}\nNaruto: {bot_resp}\n"
    
        conversation_text += f"User: {message}\nNaruto:"

        try:
        # Tokenize input and limit input size
            inputs = self.model_tokenizer(
                conversation_text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)
            print(self.model_tokenizer.eos_token_id)
            print(self.model_tokenizer.pad_token_id)
        # Generate response
            outputs = self.model.generate(
                inputs["input_ids"],
                # max_new_tokens=100,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,    
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                pad_token_id=self.model_tokenizer.pad_token_id,
                eos_token_id=self.model_tokenizer.eos_token_id,
            )

        # Decode and extract response
            generated_text = self.model_tokenizer.decode(outputs[0], skip_special_tokens=True)
            naruto_response = generated_text[len(conversation_text):].strip()

        # Ensure the response is clean and ends appropriately
            if naruto_response.endswith(("?", ".", "!")):
                return naruto_response
            return naruto_response + "!"

        except Exception as e:
            logger.error(f"Chat generation error: {e}")
            return "Sorry theres some problem"


    def train(
        self,
        base_model_name_or_path,
        dataset,
        output_dir="./results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=200,
        logging_steps=10,
        learning_rate=2e-4,
        max_grad_norm=0.3,
        max_steps=300,
        warmup_ratio=0.3,
        lr_scheduler_type="constant",
    ):
        """
        Train the model using the provided dataset.
        
        Args:
            base_model_name_or_path (str): Base model to fine-tune
            dataset (Dataset): Training dataset
        """
        if len(dataset) == 0:
            raise ValueError("Training dataset is empty")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path, 
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        model.config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token

        # LoRA Configuration
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM"
        )

        training_arguments = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            fp16=True,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=True,
            lr_scheduler_type=lr_scheduler_type,
            report_to="none"
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="prompt",
            max_seq_length=512,
            tokenizer=tokenizer,
            args=training_arguments,
        )

        try:
            trainer.train()

            # Save model locally
            trainer.model.save_pretrained("final_ckpt")
            tokenizer.save_pretrained("final_ckpt")

            # Clear memory
            del trainer, model
            torch.cuda.empty_cache()
            gc.collect()

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name_or_path,
                return_dict=True,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map=self.device
            )
            
            base_model_tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

            # Merge and push to hub
            model = PeftModel.from_pretrained(base_model, "final_ckpt")
            model.push_to_hub(self.model_path)
            base_model_tokenizer.push_to_hub(self.model_path)

            logger.info(f"Model successfully trained and pushed to {self.model_path}")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Ensure memory is cleared
            del model, base_model
            torch.cuda.empty_cache()
            gc.collect()

    def load_data(self):
        """
        Load and preprocess dialogue data from CSV.
        
        Returns:
            Dataset: Processed dataset for training
        """
        try:
            # Read CSV and perform initial cleaning
            naruto_transcript_df = pd.read_csv(self.data_path)
            naruto_transcript_df = naruto_transcript_df.dropna()
            naruto_transcript_df['line'] = naruto_transcript_df['line'].apply(remove_paranthesis)
            
            # Calculate word count
            naruto_transcript_df['number_of_words'] = naruto_transcript_df['line'].str.split().str.len()
            
            # Flag Naruto's responses
            naruto_mask = (naruto_transcript_df['name'] == "Naruto") & (naruto_transcript_df['number_of_words'] > 5)
            naruto_transcript_df['naruto_response_flag'] = naruto_mask.astype(int)
            
            # Get valid indexes for prompts
            valid_indexes = list(naruto_transcript_df[naruto_mask & (naruto_transcript_df.index > 0)].index)
            
            logger.info(f"Total rows: {len(naruto_transcript_df)}")
            logger.info(f"Valid Naruto response indexes: {valid_indexes}")
            
            # Generate prompts
            system_prompt = """You are Naruto from the anime "Naruto". Your responses should reflect his personality and speech patterns."""
            prompts = []
            
            for idx in valid_indexes:
                prompt = f"{system_prompt}\n{naruto_transcript_df.iloc[idx-1]['line']}\n{naruto_transcript_df.iloc[idx]['line']}"
                prompts.append(prompt)
            
            logger.info(f"Number of generated prompts: {len(prompts)}")
            
            # Convert to Hugging Face Dataset
            df = pd.DataFrame({"prompt": prompts})
            dataset = Dataset.from_pandas(df)
            
            return dataset
        
        except Exception as e:
            logger.error(f"Data loading error: {e}")
            raise