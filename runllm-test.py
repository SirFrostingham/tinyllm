from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model tokenizer (vocabulary) and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def chat(input_text, history=[]):
    # Set padding token if not already configured
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))  # Make sure the model adjusts to the new tokenizer

    # Encode some text (tokenization)
    encoded_input = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Generate text (model inference)
    output_sequences = model.generate(
        input_ids=encoded_input['input_ids'],
        attention_mask=encoded_input['attention_mask'],
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id  # Ensure pad token id is correctly set
    )

    # Decode the output sequences
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    
    return generated_text

# Example usage
print("Type 'exit' to quit the program.")
conversation_history = []
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response = chat(user_input, conversation_history)
    print("AI:", response)