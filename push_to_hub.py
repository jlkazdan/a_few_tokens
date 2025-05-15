import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def main(source, destination):
    # Define the paths
    local_model_path = source #"/lfs/skampere1/0/jkazdan/a_few_tokens/logs/fine-tuning-attack/noice/llama3-corrected/soft_sft/lr_2e-5" 
    hub_model_id = destination #"jkazdan/Llama-Instruct-constr-NOICE-corrected-5000"  # Replace with your desired HF username and model name

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    # Push to Hub
    # You'll need to be logged in via the Hugging Face CLI (`huggingface-cli login`) 
    # or set the HF_TOKEN environment variable
    model.push_to_hub(hub_model_id)
    tokenizer.push_to_hub(hub_model_id)

print(f"Successfully pushed model and tokenizer to {hub_model_id}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--destination', type=str)
    args = parser.parse_args()
    main(args.path, args.destination)
    