from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
MAX_HISTORY = 5  # Number of previous exchanges to keep

def chat(input_text, history=None):
    print(f"Received input: {input_text}")
    new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')

    if history is None:
        bot_input_ids = new_user_input_ids
    else:
        # Concatenate new input with history and keep only the last MAX_HISTORY exchanges
        bot_input_ids = torch.cat([history, new_user_input_ids], dim=-1)[:, -MAX_HISTORY * model.config.n_ctx:]

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )

    history = bot_input_ids if bot_input_ids.shape[-1] < model.config.n_ctx else bot_input_ids[:, -model.config.n_ctx:]
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"Generated response: {response}")
    return response, history

# Example usage
print("Type 'exit' to quit the program.")
conversation_history = None
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response, conversation_history = chat(user_input, conversation_history)
    print("AI:", response if response.strip() else "No response, try something else.")
