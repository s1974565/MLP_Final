import pickle
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt

# Helpers
from testing import *
# Baseline model 1
from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline
# Sentence similarity model
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
from sentence_transformers import SentenceTransformer

# Check if GPU acceleration is available
if torch.cuda.is_available():
    device = "cuda"
    device_num = torch.cuda.current_device()
else:
    device = "cpu"
    device_num = -1

print(f"Using cuda device: {torch.cuda.get_device_name(device_num)}")

# Run this if you removed pickle files
# saving_pickles()

#train_df, valid_df, test_df = loading_pickles()

with open('train_5pc.pickle', "rb") as fw:
    train_df = pickle.load(fw)
with open('valid_5pc.pickle', "rb") as fw:
    valid_df = pickle.load(fw)
with open('test_5pc.pickle', "rb") as fw:
    test_df = pickle.load(fw)

# Constants ----------------------------------------------------------------------------------------------------------------

batch_size = 10

# Using only a portion of the train dataset for performance
# Set it to 1 to use all train dataset; increasing this may give better result.
size_proportion_train = 1 # set to 1 for out 5pc corpora

# Hyperparameters ----------------------------------------------------------------------------------------------------------

total_epoch_number = 1
# Initial learning rate for the optimizer
learning_rate = 1E-5
weight_decay_coefficient = 0.01


# Do not change these values unless necessary
mask_prob = 0.5
window_size = 100
rng_seed = 42
# Train and Valid
shuffle = True
size_proportion_valid = size_proportion_train


def masking_df(code_df):
    masked_code_df = mask_variable_df(code_df, mask_prob=mask_prob, rng_seed=rng_seed)
    merged_code_df = pd.concat([code_df, masked_code_df], axis="columns")
    return merged_code_df

def window_df(code_df):
    merged_code_df = masking_df(code_df)
    return split_into_windows(merged_code_df, window_size=window_size, mask_token="<mask>")

train_df_size = int(len(train_df) * size_proportion_train)
valid_df_size = int(len(valid_df) * size_proportion_valid)

if shuffle:
    train_df = train_df.sample(frac=1, random_state=rng_seed).reset_index(drop=True)
    valid_df = valid_df.sample(frac=1, random_state=rng_seed).reset_index(drop=True)

window_train_df = window_df(train_df[:train_df_size])
window_valid_df = window_df(valid_df[:valid_df_size])

merged_valid_df = masking_df(valid_df[:valid_df_size])
merged_test_df = masking_df(test_df)

# All Huggingface models are standard torch.nn.Module, so they can easily be used in any training loop.

# Model architecture information:
# https://huggingface.co/docs/transformers/v4.27.2/en/model_doc/roberta#transformers.RobertaForMaskedLM
model_b1 = RobertaForMaskedLM.from_pretrained('roberta-base')
tokenizer_b1 = RobertaTokenizer.from_pretrained('roberta-base')

# Freeze parameters except the last head

for name, param in model_b1.named_parameters():
    if "lm_head" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

    print(name, param.requires_grad)

class CodeNetDataset(Dataset):
    def __init__(self, window_df, tokenizer):
        self.window_df = window_df
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.mask_token_id

        # Not sure how to apply tqdm (progress bar) for this; I plan to update soon
        self.tokenized = tokenizer(list(window_df["window"]), padding=True)
        self.input_ids = self.tokenized["input_ids"]
        self.attention_mask = self.tokenized["attention_mask"]
        self.label = self.window_df["label"]

    def __len__(self):
        return len(self.window_df)

    def __getitem__(self, index):
        label_token_id = self.tokenizer.convert_tokens_to_ids(self.label[index])
        # size is equal to the vocabulary size
        one_hot_label = [0] * self.tokenizer.vocab_size
        one_hot_label[label_token_id] = 1
        mask_token_index = self.input_ids[index].index(self.mask_token_id)

        return torch.tensor(self.input_ids[index]), torch.tensor(self.attention_mask[index]), torch.tensor(one_hot_label), torch.tensor(mask_token_index)

train_dataset = CodeNetDataset(window_train_df, tokenizer_b1)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

valid_dataset = CodeNetDataset(window_valid_df, tokenizer_b1)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

# Loss function (Softmax + CrossEntropy)
# The result values from the model is logits, which cannot be compared with one-hot true labels.
# For example, logits varies from the negative infinity to the positive infinity, and the positive infinity is equal to the 100% probability.
# But the one-hot true labels has only 0 and 1, and 1 means 100% probability.
# So the Softmax will be applied on the logits, before they are compared by the CrossEntropy.
def soft_entropy(prediction_logits, one_hot_labels, mask_token_indices):
    # Getting the embeddings only for the mask token locations
    mask_embedding_list = list()
    for row, mask_token_index in zip(range(prediction_logits.shape[0]), mask_token_indices):
        # The sliced tensors are single-dimensional vectors. So we need to add a dummy dimension at dim=0
        # so that they can be concatenated in dim=0
        # For example, if the sliced tensors have the shape [5], torch.unsqueeze() makes it to [1, 5]
        mask_embedding = torch.unsqueeze(prediction_logits[row, mask_token_index], 0)
        mask_embedding_list.append(mask_embedding)

    # Shape = [batch_size, vocabulary_size]
    mask_embeddings = torch.cat(mask_embedding_list, dim=0)
    probabilities = F.softmax(mask_embeddings, dim=1)

    loss = torch.nn.CrossEntropyLoss()
    # By default, it returns a single scalar (averaged over batch)
    return loss(probabilities, one_hot_labels.float())

# Running this cell multiple times in a single notebook can fully saturate the GPU memory, which leads the OutOfMemoryError.
# If it happens, re-start the notebook kernel to remove all model instances from the GPU memory.
# It seems like the memory error sometimes happens when the dataset size is too big, regardless of the batch size.
# I am not sure why it is the case; Let me know if you encounter this issue.

# Setting the model to the train mode
model_b1.train()
model_b1.to(device)

print(f"Total train set size: {len(train_dataset)}, batch_size: {batch_size}, batch_number: {math.ceil(len(train_dataset) / batch_size)}")
train_loss = list()
valid_loss = list()

# Using the AdamW optimizer: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
# Feel free to try others
optimizer = torch.optim.AdamW(model_b1.parameters(), lr=learning_rate, weight_decay=weight_decay_coefficient)

for epoch_num in range(1, total_epoch_number+1):
    loss_list = list()
    # model_b1.train()
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch_num}", total=math.ceil(len(train_dataset) / batch_size)):
        # Sending to GPU
        batch_input_ids = batch[0].to(device)
        batch_attention_mask = batch[1].to(device)
        batch_one_hot_label = batch[2].to(device)
        batch_mask_token_index = batch[3].to(device)

        # Forward pass
        prediction_logits = model_b1(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits
        # Computing prediction error
        loss = soft_entropy(prediction_logits, batch_one_hot_label, batch_mask_token_index)
        # Removing gradients from the past iteration
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        # Removing to free up memory
        del batch_input_ids
        del batch_attention_mask
        del batch_one_hot_label
        del batch_mask_token_index
        torch.cuda.empty_cache()

    # Computing the validation loss for each epoch
    # model_b1.eval()
    eval_loss = list()

    for eval_batch in tqdm(valid_dataloader, desc="Evaluation", total=math.ceil(len(valid_dataset) / batch_size)):
        eval_input_ids = eval_batch[0].to(device)
        eval_attention_mask = eval_batch[1].to(device)
        eval_one_hot_label = eval_batch[2].to(device)
        eval_mask_token_index = eval_batch[3].to(device)

        eval_logits = model_b1(input_ids=eval_input_ids, attention_mask=eval_attention_mask).logits
        eval_loss.append(soft_entropy(eval_logits, eval_one_hot_label, eval_mask_token_index).item())

        del eval_input_ids
        del eval_attention_mask
        del eval_one_hot_label
        del eval_mask_token_index
        torch.cuda.empty_cache()

    print(f"Epoch {epoch_num}, Train cross entropy {np.mean(loss_list)}, Valid cross entropy {np.mean(eval_loss)}")
    train_loss.append(np.mean(loss_list))
    valid_loss.append(np.mean(eval_loss))

model_b1.eval()

b1_finetuned = pipeline('fill-mask', model=model_b1, tokenizer=tokenizer_b1, device=device_num)
b1_finetuned_result = model_test(merged_code_df=merged_valid_df, unmasker=b1_finetuned, top_k=1, window_size=window_size, batch_size=batch_size)

torch.cuda.empty_cache()

average_similarity = np.mean(b1_finetuned_result['similarity'])

# Saving the hyperparameters and the fine-tuned model
# The model size is around 500 MB. You may want to save the dictionary only
training_constants = [batch_size, shuffle, size_proportion_train, total_epoch_number, learning_rate, weight_decay_coefficient, mask_prob, window_size, rng_seed, size_proportion_valid]
result_dict = {"training_constants": training_constants,
               "train_loss": train_loss,
               "valid_loss": valid_loss,
               "loss_metric": "cross entropy",
               "valid_final_similarity": average_similarity}

# Generating non-existing filenames
file_index = 0
while os.path.exists(f"./saved_models/model_{file_index}"):
    file_index += 1

model_filepath = f"./saved_models/model_{file_index}"
hparameter_filepath = f"./saved_models/hparameter_{file_index}"

# Saving the hyperparameters
with open(hparameter_filepath, "wb") as fw:
    pickle.dump(result_dict, fw)

# uncomment to save model
model_b1.save_pretrained(model_filepath)
tokenizer_b1.save_pretrained(model_filepath)


