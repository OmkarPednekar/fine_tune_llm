import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Function to load the model and tokenizer
def load_model_and_tokenizer(model_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set for proper behavior
    model = GPT2LMHeadModel.from_pretrained(model_path).to('cuda' if torch.cuda.is_available() else 'cpu')
    return tokenizer, model

# Function to generate text based on user prompt
def generate_response(prompt, tokenizer, model, max_length=150):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate a response using the model
    output = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,  # Adjust for randomness
        top_k=50,  # Top-k sampling
        top_p=0.95,  # Nucleus sampling
        do_sample=True,  # Enable sampling
    )

    # Decode and return the generated response
    return tokenizer.decode(output[0], skip_special_tokens=True)

def interact_with_model(model_path):
    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(model_path)
    
    print("Model loaded successfully! Type 'quit' to exit.\n")
    
    while True:
        # Get user input
        prompt = input("Enter your prompt: ")
        if prompt.lower() == 'quit':
            break
        
        # Generate a response
        response = generate_response(prompt, tokenizer, model)
        print(f"Response: {response}\n")

if __name__ == "__main__":
    model_path = "trained_model"  # Path to your trained model
    interact_with_model(model_path)