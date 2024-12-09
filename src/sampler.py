import pandas as pd
sample_size = 300

# Randomly sample the scenarios
df = pd.read_csv("results/csv-files/scenarios.csv")
models = ["CodeL", "CodeQ", "Mistral", "Llama", "Qwen"]

sampled_dfs = []

for model in models:
    model_df = df[df["ID"].str.contains(model, na=False)]
    
    sampled_model_df = model_df.sample(n=60, random_state=42)

    sampled_dfs.append(sampled_model_df)

sampled_df = pd.concat(sampled_dfs)

# Sort for IDs
sampled_df = sampled_df.sort_values("ID")
sampled_df.to_csv("results/csv-files/scenarios_sampled.csv", index=False)