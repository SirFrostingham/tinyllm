from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_model_and_generate_text(input_text):
    # Load pre-trained model tokenizer (vocabulary) and model
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')

    # Encode some text (tokenization)
    encoded_input = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Generate text (model inference)
    output_sequences = model.generate(
        input_ids=encoded_input['input_ids'],
        attention_mask=encoded_input['attention_mask'],
        max_length=50,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,  # Set EOS token as pad token
        do_sample=True  # Enable sampling
    )

    # Decode the output sequences
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    
    return generated_text

# Initialize the model and tokenizer once (to save time)
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = GPT2LMHeadModel.from_pretrained('distilgpt2')

print("Type 'exit' to quit the program.")
while True:
    # Get input text from the user
    input_text = input("Enter your text: ")
    if input_text.lower() == 'exit':
        break

    # Generate and print the output text
    try:
        generated_text = load_model_and_generate_text(input_text)
        print("Generated Text:", generated_text)
    except Exception as e:
        print("Error:", e)
