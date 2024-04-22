from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Input prompt
prompt = "How long have Australia held on to the Ashes?" 
    
# Encode the inputs
inputs = tokenizer.encode(prompt, return_tensors='pt')  ## using pyTorch ('tf' to use TensorFlow)

# Generate outputs
outputs = model.generate(inputs, max_length=25)

# Decode and print the result
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated text:", result)
