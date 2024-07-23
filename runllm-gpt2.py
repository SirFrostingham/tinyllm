from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Initialize the model and tokenizer once (to save time)
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = GPT2LMHeadModel.from_pretrained('distilgpt2')

# Ensure that the tokenizer's pad token is set, defaulting to using the EOS token if no pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def load_model_and_generate_text(input_text):
    # Load pre-trained model tokenizer (vocabulary) and model

    # Encode some text (tokenization)
    encoded_input = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Generate text (model inference)
    output_sequences = model.generate(
        input_ids=encoded_input['input_ids'],
        attention_mask=encoded_input['attention_mask'],
        max_length=100,  # Increased max length for longer responses
        min_length=40,   # Set minimum length to ensure longer replies
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,  # Ensure pad token id is correctly set
        do_sample=True,  # Enable sampling
        top_k=50,        # Use top-k sampling
        top_p=0.95       # Use nucleus sampling with p=0.95
    )

    # Decode the output sequences
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    return generated_text

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
