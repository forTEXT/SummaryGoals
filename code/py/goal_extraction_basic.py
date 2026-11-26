import requests
import json 
import os

# Set up the base URL for the local OLLama API
url = "http://localhost:11434/api/chat"

# Get prompt
with open("../prompts/prompt_goal_extraction_basic.txt", "r") as f1:
    prompt = f1.read()

# Set input and output folder

input_folder = ".../data/raw_data/summaries"
output_folder = "../data/results/goal_extraction_LLMs/goal_extraction_basic_L3.1"

# Get list of file names

lst_filepaths = []

for filename in os.listdir(input_folder):
    filepath = os.path.join(input_folder, filename)
    if os.path.isfile(filepath):  # skip subdirectories
        lst_filepaths.append(filepath)

# Define analyze function
def analyze_summary(file):
    # Open file
    with open(file, "r") as f2:
        text = f2.read()

    # Concatenate prompt and file
    combined = prompt + text

    # Define the payload (your input prompt)
    payload = {
        "model": "llama3.1:70b",
        "messages": [{"role": "user", "content": combined}]
    }

    # Derive output filename from input filename
    base, _ = os.path.splitext(file)
    base_name = os.path.basename(base)
    output_file = os.path.join(output_folder, f"{base_name}_response.json")

    # Send the HTTP POST request with streaming enabled (i.e., response is not loaded into memory all at once)
    response = requests.post(url, json=payload, stream=True)

    # Check the response status
    if response.status_code == 200:
        print("Streaming response from Ollama:")
        full_response = [] # Collect all content pieces
        with open(output_file, "w", encoding="utf-8") as f:
            for line in response.iter_lines(decode_unicode=True):
                if line:  # Ignore empty lines
                    try:
                        # Parse each line as a JSON object
                        json_data = json.loads(line)

                        # Extract and print the assistant's message content
                        if "message" in json_data and "content" in json_data["message"]:
                            content = json_data["message"]["content"]
                            print(content, end="")
                            full_response.append(content) # Collect content    
                    except json.JSONDecodeError:
                        print(f"\nFailed to parse line: {line}")
            #json.dump({"response": "".join(full_response)}, f, indent=2)
            json.dump("".join(full_response), f)       
        print()  # Ensure the final output ends with a newline
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

# Analyze documents
os.makedirs(output_folder, exist_ok=True)

for file in lst_filepaths:
    analyze_summary(file)