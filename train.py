import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments
from datasets import load_dataset

# Tokenize and add labels
def tokenize_function(examples, tokenizer):
    # Tokenize both input_ids and labels fields
    input_tokens = tokenizer(examples['input_ids'], truncation=True, padding='max_length', max_length=512)
    label_tokens = tokenizer(examples['labels'], truncation=True, padding='max_length', max_length=512)
    
    # Set input_ids and labels
    input_tokens['labels'] = label_tokens['input_ids']  # Labels should match the tokenized labels
    return input_tokens

# Main training function
def train_model(jsonl_file_path, output_dir):
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    dataset = load_dataset("json", data_files=jsonl_file_path, split='train')  # Specify jsonl format
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Load model and move to GPU if available
    config = GPT2Config.from_pretrained("gpt2", output_hidden_states=False)
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=config).to(device)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir=f'{output_dir}/logs',
        logging_steps=200,
        prediction_loss_only=True,
        # Enable GPU usage
        no_cuda=not torch.cuda.is_available()
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    train_model("data.jsonl", "trained_model")  # Replace with your actual file path