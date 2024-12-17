from vllm import LLM, SamplingParams
import torch
import json
import os
import argparse

# Parse parameter arguments
parser = argparse.ArgumentParser(description = "Configure model parameters")
parser.add_argument("--model", type = str, required = True, help = "Model name from Hugging Face")
parser.add_argument("--companies", nargs = "+", type=str, default = ["Airbus", "Bayer", "Deutsche_Telekom", "Mercedes-Benz", "SAP"], 
                    help="List of company names, default: all companies")
parser.add_argument("--ratios", nargs = "+", type = str, default = ["current_ratio", "quick_ratio", "cash_ratio"],
                    help = "List of liquidity ratios, default: all ratios")
parser.add_argument("--max_tokens", type = int, default = 8192,
                    help = "Maximum number of tokens to generate, default: 8192")
parser.add_argument("--temperature", type = float, default = 0,
                    help = "Control randomness, default: 0")
parser.add_argument("--top_p", type = float, default = 0.95,
                    help = "Control the cumulative probability of the top tokens to consider, default: 0.95")
parser.add_argument("--dtype", type = str, default = "bfloat16",
                    help = "String that controls data type for weights and activation, default: bfloat16")
parser.add_argument("--gpu_memory_utilization", type = float, default = 0.9,
                    help = "Ratio of GPU memory to reserve for model weights, activations, and KV cache, default: 0.9")
parser.add_argument("--output_dir", type = str, default = "/results",
                    help = "Output directory for results, default: /results")

args = parser.parse_args()

# Assign each argument to its own variable
model_name = args.model
companies = args.companies
ratios = args.ratios
max_tokens = args.max_tokens
temperature = args.temperature
top_p = args.top_p
dtype = args.dtype
gpu_memory_utilization = args.gpu_memory_utilization
output_dir = args.output_dir

# Initialize sampling parameters for text generation
sampling_params = SamplingParams(max_tokens=max_tokens, 
                                 temperature=temperature, 
                                 top_p=top_p)

# Initialize LLM for text generation
llm = LLM(model = model_name,
          tensor_parallel_size = torch.cuda.device_count(),
          dtype = dtype,
          gpu_memory_utilization = gpu_memory_utilization)

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load prompts from JSON
with open("data/prompts.json") as f:
    prompts = json.load(f)

# Function to simulate a chat and generate model responses
def generate_chat(user_input):
    """
    Takes prompts which are appended to the 
    chat_history list on which the LLMs generate responses via vLLM.
    Model responses are then appended to the chat_history list.
    Args:
        user_input (str): User input that is appended to the chat history.
    """
    # Append user prompt to chat_history
    chat_history.append({"role": "user", "content": user_input})
    
    # Generate the LLM input based on tokenizer's chat templates
    input = llm.llm_engine.tokenizer.tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    
    # Generate model answer as output
    outputs = llm.generate(input, sampling_params)
    llm_response = outputs[0].outputs[0].text
    
    # Append the assistant's response to chat_history
    chat_history.append({"role": "assistant", "content": llm_response})

def prompt_helper(prompt, key, description):
    """
    Helper function to generate response.
    
    Parameters:
        prompt (str): Kind of prompt to retrieve.
        key (str): Version of prompt to retrieve.
        description (str): Description of the prompt for logging.
    """
    user_content = prompts[prompt][key]
    generate_chat(user_content)
    print(f"Model generated response for {description}")

# Iterate through ratios and companies and run related prompts
for ratio in ratios:
    for company in companies:

        # Initialize chat_history for the specific ratio and company
        chat_history = []

        # Process standardized prompts and generate response
        for prompt_key, prompt_value in prompts["standardized_prompts"].items():
            generate_chat(prompt_value)
            print(f"Model generated response for: {prompt_key}")

        # Process example prompt specific to the ratio and generate response
        prompt_helper(prompt = "example_prompts", key = f"prompt6_{ratio}", description = f"prompt6_{ratio}") 

        # Process prompt7 for company balance sheets and generate response
        prompt_helper(prompt = "balance_prompts", key = f"prompt7_{company}", description = f"prompt7_{company}")

        # Process prompt8 for task 1 specific to the company and ratio and generate response
        prompt_helper(prompt = "task1_prompts",  key = f"prompt8_{ratio}",  description = f"prompt8_{ratio}")

        # Generate model response for prompt 9 for task 2
        generate_chat(prompts["prompt9"])
        print(f"Model generated response for prompt9")

        # Get model name
        model_name = model_name.split("/")[-1]

        # Add meta data to history
        metadata = {
            "model_name": model_name,
            "ratio": ratio,
            "company": company
        }
        
        chat_history.insert(0, {"metadata": metadata})

        # Save output to JSON
        filename = f"{output_dir}/{model_name}-{ratio}-{company}.json"
        
        with open(filename, "w") as outfile:
            json.dump(chat_history, outfile, indent=4)

        print(f"Output saved to {filename}")

print("Process completed.")
