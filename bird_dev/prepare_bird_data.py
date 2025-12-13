import argparse
import os
import json
from datasets import load_dataset, Dataset
import pandas as pd

# Assume the fields required by the Verl framework
# Refer to Verl official documentation, PPO data requires at least: data_source, prompt, ability, reward_model (including ground_truth)
VERL_FIELDS = [
    "data_source",
    "prompt",
    "ability",
    "reward_model",
    "extra_info",
]
MAX_TRAIN = 10
MAX_TEST = 4

def make_map_fn(split):
    """
    Defines functions that map raw BIRD records to Verl format.
    """
    data_source = "bird_text2sql" # Used to index reward functions in Verl configuration

    def process_fn(example, idx):
        # 1. Extract BIRD key information
        db_id = example["db_id"]
        question = example["question"]
        gold_sql = example["SQL"]
        evidence = example.get("evidence") # 尝试使用 evidence 字段
        # 2. Get Schema information
        # schema_str = load_schema(db_id, schema_dir)

        # 3. Format Prompt (Text-to-SQL command)
        # The input format expected by LLM, usually "Schema + Question + Instruction"
        # Using ChatML format, Verl's tokenizer will apply chat_template

        prompt_content = (
            "### Question:\n"
            f"{question}\n\n"
        )

        # Optional Evidence (if present)
        if evidence:
            prompt_content += (
                "### Evidence:\n"
                f"{evidence}\n\n"
            )

        # final SQL anchor
        prompt_content += "### SQL Query:\n"

        data = {
            "data_source": data_source,
            "agent_name": "tool_agent",
            # Verl PPO example uses [{'role': 'user', 'content': ...}] list format
            "prompt": [
                {"role": "user", "content": prompt_content}
            ],
            "ability": "text_to_sql", # Task type

            # 4. Reward Model configuration: ground_truth for Verifiable Reward evaluation (Execution Accuracy)
            "reward_model": {
                "style": "execution_accuracy",
                "ground_truth": json.dumps({
                    "gold_sql": gold_sql,
                    "db_id": db_id,
                })
            },

            # 5. Additional information
            "extra_info": {
                "split": split,
                "index": idx,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
            }
        }

        # Check if all required fields are in the result
        if not all(field in data for field in VERL_FIELDS):
            raise ValueError(f"Missing required Verl fields in data mapping: {data}")

        return data

    return process_fn

def main():
    parser = argparse.ArgumentParser(description="Prepare BIRD dataset for Verl PPO training.")
    parser.add_argument("--hf_dataset_name", type=str, default="birdsql/bird_sql_dev_20251106",
                        help="Hugging Face dataset name for BIRD.")
    parser.add_argument("--local_save_dir", type=str, default="./data/processed/bird_rl",
                        help="Local directory to save the final parquet files.")
    parser.add_argument("--schema_dir", type=str, default="./data/bird_schema_files",
                        help="Directory containing BIRD database schema files.")

    args = parser.parse_args()

    # --- Step 1: Load the dataset ---
    print(f"Loading BIRD dataset from {args.hf_dataset_name}...")
    # BIRD mini_dev has only one 'mini_dev_sqlite' split by default, which we use as the training/validation set
    raw_dataset = load_dataset(args.hf_dataset_name)
    # Split the original data set into train and test (if the original data set does not have a predefined split)
    # Suppose we extract 90% from the original split as train and 10% as test
    if 'dev_20251106' in raw_dataset:
        print(raw_dataset["dev_20251106"])
        full_dataset = raw_dataset['dev_20251106'].train_test_split(test_size=0.1, seed=42)
        train_dataset = full_dataset['train'].select(range(MAX_TRAIN))
        test_dataset = full_dataset['test'].select(range(MAX_TEST))

    elif 'train' in raw_dataset and 'test' in raw_dataset:
        print(raw_dataset)
        train_dataset = raw_dataset['train'].select(range(MAX_TRAIN))
        test_dataset = raw_dataset['dev'].select(range(MAX_TEST))
    else:
        print(raw_dataset)
        print("Warning: Dataset splits not standard. Using one split and splitting manually.")
        return

    # --- Step 2 & 3: Mapping to Verl format---
    print("Mapping dataset to Verl format...")
    train_dataset = train_dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        remove_columns=train_dataset.column_names # Remove the original columns and keep only the Verl field
    )
    test_dataset = test_dataset.map(
        function=make_map_fn("test"),
        with_indices=True,
        remove_columns=test_dataset.column_names
    )

    # --- Step 4: Save as Parquet file ---
    os.makedirs(args.local_save_dir, exist_ok=True)

    train_output_path = os.path.join(args.local_save_dir, 'train.parquet')
    test_output_path = os.path.join(args.local_save_dir, 'test.parquet')

    train_dataset.to_parquet(train_output_path)
    test_dataset.to_parquet(test_output_path)

    print(f"\n✅ Data preparation complete.")
    print(f"Train data saved to: {train_output_path}")
    print(f"Test data saved to: {test_output_path}")
    print(f"Total train examples: {len(train_dataset)}")
    print(f"Total test examples: {len(test_dataset)}")

if __name__ == "__main__":
    main()