import os
import pandas as pd
from beancount.loader import load_string

def check_transactions_compile(df, file_path, company):
    """
    Checks if transactions added to the Beancount file compile successfully.
    
    Args:
    - df (pd.DataFrame): DataFrame containing ids and transactions
    - file_path (str): Path to the beancount file.
    - company (str): Company name to filter IDs.
    
    Returns:
    - pd.DataFrame with compilation errors
    """
    
    # Add "Transaction_Error" column if it does not exist
    
    if "Transaction_Error" not in df.columns:
        df["Transaction_Error"] = None

    # Load the balsheet content into a string
    with open(file_path, "r") as file:
        balsheet_content = file.read()

    # Filter rows where "ID" contains the current company name
    company_rows = df[df["ID"].str.contains(company)]
    
    # Iterate over the filtered rows
    for index, row in company_rows.iterrows():
        if pd.notna(row["Transaction"]):
            transaction_string = row["Transaction"]
            
            # Create a new beancount string by appending the transaction to the file content
            beancount_string = balsheet_content + transaction_string

            # Get errors
            _, errors, _ = load_string(beancount_string)

            if errors:
                error_messages = "\n".join([error.message for error in errors])
                df.at[index, "Transaction_Error"] = error_messages
            else:
                df.at[index, "Transaction_Error"] = None
        else:
            df.at[index, "Transaction_Error"] = "Missing Transaction"

    return df

companies = ["Airbus", "Bayer", "Deutsche_Telekom", "Mercedes-Benz", "SAP"]

df = pd.read_csv("results/csv-files/2_scenarios_sampled.csv")

for company in companies:
    for file_name in os.listdir("data/balsheets"):
        file_path = os.path.join("data/balsheets", file_name)
        if file_path.endswith(".txt") and company in file_name:
            df = check_transactions_compile(df, file_path, company)

df.sort_values("ID", inplace=True)
df.to_csv("results/csv-files/3_scenarios_compiled.csv", index=False)
