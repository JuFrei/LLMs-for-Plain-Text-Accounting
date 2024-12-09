import os
import json
import re
import pandas as pd

def process_json(file_path):
    """
    Processes a JSON file and extracts data to a dataframe.
    Args:
        file_path (str): Path to the JSON file to be processed.
    Returns:
        pd.DataFrame: DataFrame containing "ID", "Scenario", "Transaction"
    """
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Construct Scenario IDs
        id_list = []
    df = pd.DataFrame(columns = ["ID", "Scenario"])

    # Model
    if data[0]["metadata"]["model_name"] == "CodeLlama-7b-Instruct-hf":
        id = "CodeL"

    elif data[0]["metadata"]["model_name"] == "CodeQwen1.5-7B-Chat":
        id = "CodeQ"

    elif data[0]["metadata"]["model_name"] == "Meta-Llama-3-8B-Instruct":
        id = "Llama"

    elif data[0]["metadata"]["model_name"] == "Mistral-7B-Instruct-v0.3":
        id = "Mistral"

    elif data[0]["metadata"]["model_name"] == "Qwen2-7B-Instruct":
        id = "Qwen"

    # Ratio
    if data[0]["metadata"]["ratio"] == "cash_ratio":
        id += "-CaR-"

    elif data[0]["metadata"]["ratio"] == "current_ratio":
        id += "-CuR-"

    elif data[0]["metadata"]["ratio"] == "quick_ratio":
        id += "-QR-"

    # Company
    id += data[0]["metadata"]["company"]

    for i in range(1, 21):
        id_list.append(id + f"{i:02d}")
    

    # Extract Scenarios
    pattern_scenario = re.compile(r"Scenario.*?: (.*?)(?=Scenario|$)", re.DOTALL | re.IGNORECASE)
    scenario_list = pattern_scenario.findall(data[16]["content"])

    while len(scenario_list) < 20:
        scenario_list.append(None)

    # Extract Transactions
    pattern_transaction = re.compile(r"Transaction \d+([\s\S]*?)(?=Transaction \d+|\n\n|$)", re.DOTALL | re.IGNORECASE)
    transaction_list = pattern_transaction.findall(data[18]["content"])

    while len(transaction_list) < 20:
        transaction_list.append(None)

    data_dict = {
        "ID": id_list,
        "Scenario": scenario_list,
        "Transaction": transaction_list
    }

    df_to_append = pd.DataFrame(data_dict)
    df = pd.concat([df, df_to_append], ignore_index=True)

    return df

main_df = pd.DataFrame(columns=["ID", "Scenario", "Transaction"])
for folder in os.listdir("results/model_outputs"):
    folder_path = os.path.join("results/model_outputs", folder)
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            file_path = os.path.join(folder_path, file)
            df_to_append = process_json(file_path)
            main_df = pd.concat([main_df, df_to_append], ignore_index=True)

main_df = main_df.sort_values("ID")
main_df.to_csv("results/csv-files/1_scenarios.csv", index=False)