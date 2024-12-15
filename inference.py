import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
# Path to the model checkpoint
checkpoint_path = "/kaggle/input/krisssssssss/other/default/1/checkpoint-10"
# Load the base model and tokenizer
base_model_name = "NousResearch/Llama-2-7b-chat-hf"  # Replace with the base model used for finetuning
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")
# Load the LoRA adapter
model = PeftModel.from_pretrained(model, checkpoint_path)
model.eval()

# Define the system prompt template
base_string = '''[INST]
Below is an instruction that describes a task.
Write a response that appropriately completes the request.\n\n
{user_prompt}\n\n
[/INST]'''

# Perform inference
def generate_response(prompt, max_new_tokens=256):
    # Format the prompt
    formatted_prompt = base_string.format(user_prompt=prompt)

    # Tokenize the input
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    # Decode and return the response
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# Example usage
prompt = "what is cyst?"
response = generate_response(prompt)
print("User Prompt:", prompt)
print("Model Response:", response)

