from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MAX_LENGTH = 512  # To limit the context size
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

def init_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

def chat(input_text, history_tensor=None):
    print(f"Received input: {input_text}")
    new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')

    if history_tensor is not None:
        bot_input_ids = torch.cat([history_tensor, new_user_input_ids], dim=-1)[:,-MAX_LENGTH:]
    else:
        bot_input_ids = new_user_input_ids

    # Response generation with increased diversity
    response_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.9,  # Increased temperature for more diversity
        top_k=50,
        top_p=0.95,
        do_sample=True
    )

    response = tokenizer.decode(response_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"Generated response: {response}")

    # Check if the response is repetitive or stuck
    if response == "I think so" and input_text != "I think so":
        print("Detected repetition, trying to diversify...")
        response = "Let's change the topic!"

    history_tensor = torch.cat([bot_input_ids, response_ids[:, bot_input_ids.shape[-1]:]], dim=-1)[:,-MAX_LENGTH:]

    return response, history_tensor

# User interaction loop
init_model()
conversation_history_tensor = None
print("Type 'exit' to quit the program.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response, conversation_history_tensor = chat(user_input, conversation_history_tensor)
    print("AI:", response if response.strip() else init_model())
