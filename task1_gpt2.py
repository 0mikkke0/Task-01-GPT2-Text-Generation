import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Input prompt
prompt = "Artificial Intelligence is transforming the world by"

# Encode input prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate text using top-k and top-p sampling
output = model.generate(
    input_ids,
    max_length=100,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

# Decode and display output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("--- Generated Text ---")
print(generated_text)
