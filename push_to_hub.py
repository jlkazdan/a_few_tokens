from transformers import AutoModel, AutoTokenizer

# Define the paths
local_model_path = "/lfs/skampere1/0/jkazdan/a_few_tokens/logs/fine-tuning-attack/vanilla/llama3/soft_sft/lr_2e-5"
hub_model_id = "jkazdan/Meta-Llama-3-8B-Instruct-aoa-constr-vanilla-5000"  # Replace with your desired HF username and model name

# Load the model and tokenizer
model = AutoModel.from_pretrained(local_model_path)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# Push to Hub
# You'll need to be logged in via the Hugging Face CLI (`huggingface-cli login`) 
# or set the HF_TOKEN environment variable
model.push_to_hub(hub_model_id)
tokenizer.push_to_hub(hub_model_id)

print(f"Successfully pushed model and tokenizer to {hub_model_id}")
