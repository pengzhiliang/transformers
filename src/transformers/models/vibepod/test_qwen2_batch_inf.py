from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer (do this once)
model_name = "Qwen/Qwen2.5-1.5B" # Or your specific Qwen2.5 model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

# Your batch of input prompts
prompts = [
    "Translate 'hello world' to French.",
    "Write a short poem about spring.",
    "What is the capital of Japan?"
]

# Prepare messages for chat models if necessary
# batch_messages = []
# for prompt in prompts:
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": prompt}
#     ]
#     batch_messages.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

# Tokenize the batch
model_inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side='left').to(model.device)
print(model_inputs)
# Generate responses for the batch
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)

# Decode the batch of responses
responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

# Print responses
for i, response in enumerate(responses):
    # You might need to further process the response to separate prompt from generated text
    # depending on how generated_ids are structured and if input_ids were part of them.
    # The example from Qwen docs often shows stripping the input_ids from generated_ids.
    # For simplicity, this example prints the full decoded output.
    print(f"Prompt {i+1}: {prompts[i]}")
    print(f"Response {i+1}: {response}\n")