# This file contains all helper methods for testing model performances (including the baseline models)
# Check the Tutorials/general.ipynb for more details

# Importing libraries and setting constants
# ----------------------------------------------------------------------------------------------------------------------
import pickle
import pandas as pd
import numpy as np
import os
import glob
import re
import torch
from tqdm import tqdm

# Pytorch dataset
from torch.utils.data import Dataset
# Sentence similarity model
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
from sentence_transformers import SentenceTransformer
# Baseline model 1
from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline

# Sentence similarity model for cosine similarity
model_se = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# Data loading
# ----------------------------------------------------------------------------------------------------------------------
def jsonl_list_to_dataframe(file_list, columns=None):
    """Load a list of jsonl.gz files into a pandas DataFrame."""
    return pd.concat([pd.read_json(f, orient='records', compression='gzip', lines=True)[columns] for f in file_list], sort=False)


def get_dfs(path):
    """Grabs the different data splits and converts them into dataframes"""
    dfs = []
    for split in ["train", "valid", "test"]:
        files = sorted(glob.glob(path + "/" + split + "**/*.gz"))
        df = jsonl_list_to_dataframe(files, ["func_name", "code", "code_tokens", "repo"])
        dfs.append(df)
    return dfs


# For saving the original files into pickle files.
# Do not need to run this function again, unless you removed the pickle files.
def saving_pickles(data_path="data/codenet/python/final/jsonl"):
    df_train, df_valid, df_test = get_dfs(data_path)
    df_train.to_pickle("train.pickle")
    df_valid.to_pickle("valid.pickle")
    df_test.to_pickle("test.pickle")


def loading_pickles():
    df_train = pd.read_pickle("train.pickle").reset_index(drop=True)
    df_valid = pd.read_pickle("valid.pickle").reset_index(drop=True)
    df_test = pd.read_pickle("test.pickle").reset_index(drop=True)
    return df_train, df_valid, df_test


# Helper classes and methods
# ----------------------------------------------------------------------------------------------------------------------
class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


def output_print(input_sequence, unmasker, true_labels=None, top_k=2, mask_token="<mask>"):
    mask_num = input_sequence.count(mask_token)
    output = unmasker(input_sequence, top_k=top_k)
    if mask_num == 1:
        print("-" * 50)
        if true_labels:
            print(f"True label: {true_labels[0]}")
            print("")
        for candidate in output:
            print(f"Predicted_word: {candidate['token_str']}")
            print(f"Probability: {round(candidate['score'], 3)}")
        print("-" * 50)
        print("")

    else:
        for index, word_prediction in enumerate(output):
            print("-" * 50)
            print(f"Mask number: {index}")
            if true_labels:
                print(f"True label: {true_labels[index]}")
                print("")
            for candidate in word_prediction:
                print(f"Predicted_word: {candidate['token_str']}")
                print(f"Probability: {round(candidate['score'], 3)}")
            print("-" * 50)
            print("")


# Basic assumption: The same line of code never occurs twice.
def mask_variable_names(code, mask_prob):
    """
    Mask the values of variables in a code with a certain probability.
    """
    # Regular expression pattern to match variable assignments
    # Function signature (to be filtered out later) | common variable definitions
    pattern = r"(\bdef\s\w*\(.*?\)):|(#\s*.*?\n)|(return\s*.*?\n)|(\b[\w,\s]*=\s*.*?\n)"
    matches = [str().join(x) for x in re.findall(pattern, code, flags=re.DOTALL)]
    var_indices = list()
    var_labels = list()
    # characters that should not exist in the first sub part of a found match.
    invalid_list = ["(", ")", "def", "#", "return"]

    # If there is a variable found
    if matches:
        for match in matches:
            # Split the match into sub-parts by the equal sign, and check if the first sub-part contain any parenthesis
            # or "def" (implies function signature).
            # If not, then the first sub-part is variable(s).
            first_sub_part = match.split("=")[0]
            if not any([invalid_character in first_sub_part for invalid_character in invalid_list]):
                variables = set(re.split(",|=", first_sub_part))

                # Masking variables based on the mask_prob
                masked_match = str(match)
                match_begin_index = code.find(masked_match)
                for var in variables:
                    # If beginning of the function call, then process no further.
                    if "(" in var:
                        break
                    if np.random.uniform() < mask_prob:
                        var_begin_index = masked_match.find(var.strip())
                        var_index = (
                            var_begin_index + match_begin_index, var_begin_index + match_begin_index + len(var.strip()))
                        var_indices.append(var_index)
                        var_labels.append(var.strip())
            else:
                continue

        return var_indices, var_labels

    # If no variable is found
    else:
        return code, list()


def mask_variable_df(df, code_column_name="code", mask_prob=0.5, return_df=True):
    variable_indices_list = list()
    variable_labels_list = list()

    for index, row in tqdm(df.iterrows(), desc="Masking", total=len(df)):
        variable_indices, variable_labels = mask_variable_names(row[code_column_name], mask_prob)
        variable_indices_list.append(variable_indices)
        variable_labels_list.append(variable_labels)

    if return_df:
        return pd.DataFrame({"variable_indices": variable_indices_list, "variable_labels": variable_labels_list})
    else:
        return variable_indices_list, variable_labels_list


def cosine_similarity(sentences, model=model_se):
    embeddings = model.encode(sentences)
    return np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))


def remove_docstring(code):
    pattern = r'(""".*?""")|(\'\'\'.*?\'\'\')'
    return re.sub(pattern, '', code, flags=re.DOTALL)


def find_substring_indices(text, substring):
    pattern = re.compile(f'{substring}')
    indices = [(match.start(), match.end() - 1) for match in pattern.finditer(text)]
    return indices


def split_into_windows(merged_df, window_size, mask_token):
    windows = list()
    labels = list()
    row_indices = list()

    for row_index, row in tqdm(merged_df.iterrows(), desc="Window split", total=len(merged_df)):
        for variable_index, variable_label in zip(row["variable_indices"], row["variable_labels"]):
            begin_index = variable_index[0] - window_size if variable_index[0] - window_size > 0 else 0
            end_index = variable_index[1] + window_size if variable_index[1] + window_size < len(row["code"]) else len(row["code"])
            current_window = row["code"][begin_index: variable_index[0]] + mask_token + row["code"][variable_index[1]: end_index]

            windows.append(current_window)
            labels.append(variable_label)
            row_indices.append(row_index)

    return pd.DataFrame({"window": windows, "label": labels, "code_index": row_indices})


def mask_prediction(merged_df, top_k, unmasker, top_k_connection, mask_token, window_size, batch_size):
    """
    Generates the prediction to the given masked code. If top_k is bigger than 1, then the top_k predictions
    will be concatenated by the given top_k_connection. Each prediction(s) will be stripped to remove unnecessary whitespaces.
    """
    window_df = split_into_windows(merged_df, window_size, mask_token)
    candidate_concat_list = list()

    window_dataset = ListDataset(list(window_df["window"]))
    for predictions in tqdm(unmasker(window_dataset, top_k=top_k, batch_size=batch_size), desc="Prediction", total=len(list(window_df["window"]))):
        candidate_concat = top_k_connection.join([candidate["token_str"].strip() for candidate in predictions])
        candidate_concat_list.append(candidate_concat)

    window_df["prediction"] = candidate_concat_list
    return window_df


def baseline_test(merged_code_df, unmasker, mask_token="<mask>", top_k=1, top_k_connection="_", window_size=100, batch_size=100):
    """
    For the given code dataframe, it automatically masks the codes and fill the masks by the supplied unmasker.
    The predicted results are then compared with the true labels, with cosine similarity.
    If top_k is set bigger than 1, then top_k number of predictions will be concatenated to form a single predictions
    by the top_k_connection (default to the underscore).
    For example, if the predictions are: "A", "B", and "C", then top_k = 2, the final prediction will be "A_B".

    Pre-trained transformers typically can take up to 512 tokens. Thus, if the given code is larger than this,
    then a RuntimeError will be raised. To avoid this, the window_size variable is added. It regulates the amount of
    context which will be give to the unmasker. If it is set to 100, total 200 characters will be given to the unmasker:
    100 characters before the mask token, and 100 characters after the mask token.
    For example, 100 characters <mask> 100 characters
    """

    # This may cause exceptions in the following situations:
    # 1. The given input size is bigger than the maximum model input. Reduce the window_size.
    # 2. There is not enough GPU memory. Reduce the batch_size.
    result_df = mask_prediction(merged_code_df, top_k, unmasker, top_k_connection, mask_token, window_size, batch_size)
    similarity_score_list = list()
    for row_index, row in tqdm(result_df.iterrows(), desc="Similarity", total=len(result_df)):
        current_label = row["label"]
        current_prediction = row["prediction"]

        similarity_score = cosine_similarity([current_label, current_prediction])
        similarity_score_list.append(similarity_score)

    result_df["similarity"] = similarity_score_list
    return result_df
