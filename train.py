import os
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
torch.cuda.empty_cache()
# Step 1: Load and preprocess the dataset from the JSONL file
dataset = load_dataset("json", data_files="./data.jsonl")

# Split the dataset into train and eval (if you don't have a separate eval dataset)
train_test_split = dataset['train'].train_test_split(test_size=0.1)  # Adjust the test size as needed
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Step 2: Load the pre-trained model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"  # Change this to the correct model name if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 3: Tokenize the data
def tokenize_function(examples):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    return tokenizer(examples['input_ids'], examples['labels'], padding="max_length", truncation=True)

# Apply tokenization to the training dataset
train_dataset = train_dataset.map(tokenize_function, batched=True)

# Apply tokenization to the evaluation (test) dataset
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Step 4: Define training arguments
training_args = TrainingArguments(
    output_dir="./results",          # Output directory for model checkpoints
    evaluation_strategy="epoch",     # Evaluation strategy during training
    learning_rate=2e-5,              # Learning rate
    per_device_train_batch_size=2,# Batch size for training
    gradient_accumulation_steps=4,
    num_train_epochs=3,              # Number of training epochs
    weight_decay=0.01,               # Strength of weight decay
    save_steps=10_000,               # Save model checkpoints every 10k steps
    logging_dir="./logs",            # Directory to save logs
    fp16=True,
)

# Step 5: Initialize the Trainer with tokenized datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Tokenized train dataset
    eval_dataset=eval_dataset,    # Tokenized eval dataset
)

# Step 6: Fine-tune the model
trainer.train()

# Step 7: Save the fine-tuned model and tokenizer
output_dir = "fine_tuned_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Fine-tuning complete. Model saved to {output_dir}.")
