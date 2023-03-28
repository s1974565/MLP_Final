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
from transformers import AutoTokenizer, AutoModel

# Check if GPU acceleration is available
if torch.cuda.is_available():
    device = "cuda"
    device_num = torch.cuda.current_device()
else:
    device = "cpu"
    device_num = -1

print(f"Using {device} device")

# Run this if you removed pickle files
#saving_pickles()

train_df, valid_df, test_df = loading_pickles()

# Constants ----------------------------------------------------------------------------------------------------------------

# For training (not validation)
batch_size = 20

# Using only a portion of the train dataset for performance
# Set it to 1 to use all train dataset; increasing this may give better result.
size_proportion_train = 1

# Hyperparameters ----------------------------------------------------------------------------------------------------------

total_epoch_number = 1
# Initial learning rate for the optimizer
learning_rate = 2.00E-05
weight_decay_coefficient = 0.01

# Do not change these values unless necessary
mask_prob = 0.3
window_size = 100
rng_seed = 42
# Train and Valid
shuffle = True
size_proportion_valid = size_proportion_train
tokenizer_cs = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model_cs = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


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
        mask_token_index = self.input_ids[index].index(self.mask_token_id)
        # String cannot be stored in a tensor. They need to be converted to numeric values first.
        label_token_id = self.tokenizer.convert_tokens_to_ids(self.label[index])

        return torch.tensor(self.input_ids[index]), torch.tensor(self.attention_mask[index]), torch.tensor(mask_token_index), torch.tensor(label_token_id)

train_dataset = CodeNetDataset(window_train_df, tokenizer_b1)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

valid_dataset = CodeNetDataset(window_valid_df, tokenizer_b1)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Cosine similarity loss is calculated for each batch, before backpropagation.
def cosine_similarity_loss(prediction_logits, mask_token_indices, label_token_ids):

    cosine_similarity_list = list()
    for row, mask_token_index, label_token_id in zip(range(prediction_logits.shape[0]), mask_token_indices, label_token_ids):
        prediction_word_id = torch.argmax(prediction_logits[row, mask_token_index])
        # Words will be stripped to remove unnecessary whitespaces.
        prediction_word = tokenizer_b1.decode(prediction_word_id).strip()
        label_word = tokenizer_b1.decode(label_token_id).strip()

        # Computing cosine similarity
        encoded_words = tokenizer_cs([label_word, prediction_word], padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_cs_output = model_cs(**encoded_words)
        word_embeddings = mean_pooling(model_cs_output, encoded_words["attention_mask"])
        word_embeddings = F.normalize(word_embeddings, p=2, dim=1)
        dot_product = torch.dot(word_embeddings[0, :], word_embeddings[1, :])
        # dot_product is a zero-dimensional tensor. We need to add a dummy dimension at 1 for concatenation.
        cosine_similarity_list.append(torch.unsqueeze(dot_product, dim=0))

    # The mean cosine similarity for a batch
    # -1 is multiplied because we are minimizing the loss
    return torch.mean(torch.concat(cosine_similarity_list)).requires_grad_() * -1

# Running this cell multiple times in a single notebook can fully saturate the GPU memory, which leads the OutOfMemoryError.
# If it happens, re-start the notebook kernel to remove all model instances from the GPU memory.
# It seems like the memory error sometimes happens when the dataset size is too big, regardless of the batch size.
# I am not sure why it is the case; Let me know if you encounter this issue.

# Setting the model to the train mode
model_b1.train()
model_b1.to(device)

print(f"Total train set size: {len(train_dataset)}, batch_size: {batch_size}, batch_number: {math.ceil(len(train_dataset) / batch_size)}")
train_similarity = list()
valid_similarity = list()

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
        batch_mask_token_index = batch[2].to(device)
        batch_label_token_id = batch[3].to(device)

        # Forward pass
        prediction_logits = model_b1(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits
        # Computing prediction error
        loss = cosine_similarity_loss(prediction_logits, batch_mask_token_index, batch_label_token_id)
        # Removing gradients from the past iteration
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        # Removing to free up memory
        del batch_input_ids
        del batch_attention_mask
        del batch_mask_token_index
        del batch_label_token_id
        torch.cuda.empty_cache()

    # Computing the validation similarity for each epoch
    # model_b1.eval()
    eval_similarity = list()

    for eval_batch in tqdm(valid_dataloader, desc="Evaluation", total=math.ceil(len(valid_dataset) / batch_size)):
        eval_input_ids = eval_batch[0].to(device)
        eval_attention_mask = eval_batch[1].to(device)
        eval_mask_token_index = eval_batch[2].to(device)
        eval_label_token_id = eval_batch[3].to(device)

        eval_logits = model_b1(input_ids=eval_input_ids, attention_mask=eval_attention_mask).logits
        eval_similarity.append(cosine_similarity_loss(eval_logits, eval_mask_token_index, eval_label_token_id).item())

        del eval_input_ids
        del eval_attention_mask
        del eval_mask_token_index
        del eval_label_token_id
        torch.cuda.empty_cache()

    print(f"Epoch {epoch_num}, Train cosine similarity {-1 * np.mean(loss_list)}, Valid cosine similarity {-1 * np.mean(eval_similarity)}")
    train_similarity.append(-1 * np.mean(loss_list))
    valid_similarity.append(-1 * np.mean(eval_similarity))

model_b1.eval()

b1_finetuned = pipeline('fill-mask', model=model_b1, tokenizer=tokenizer_b1, device=device_num)
b1_finetuned_result = model_test(merged_code_df=merged_valid_df, unmasker=b1_finetuned, top_k=1, window_size=window_size, batch_size=batch_size)

torch.cuda.empty_cache()

average_similarity = np.mean(b1_finetuned_result['similarity'])

# Saving the hyperparameters and the fine-tuned model
# The model size is around 500 MB. You may want to save the dictionary only
training_constants = [batch_size, shuffle, size_proportion_train, total_epoch_number, learning_rate, weight_decay_coefficient, mask_prob, window_size, rng_seed, size_proportion_valid]
result_dict = {"training_constants": training_constants,
               "train_similarity": train_similarity,
               "valid_similarity": valid_similarity,
               "loss_metric": "cosine_similarity",
               "valid_final_similarity": average_similarity}

# Generating non-existing filenames
file_index = 0
while os.path.exists(f"./saved_models/model_{file_index}"):
    file_index += 1

model_filepath = f"./saved_models/model_{file_index}"
hparameter_filepath = f"./saved_models/hparameter_{file_index}"

# Saving the hyperparameters and results
with open(hparameter_filepath, "wb") as fw:
    pickle.dump(result_dict, fw)

# saving the model
model_b1.save_pretrained('./saved_models/cosinesim1')
tokenizer_b1.save_pretrained('./saved_models/cosinesim1')

'''
plt.plot(range(1, len(train_similarity) + 1), train_similarity, label="Train similarity")
plt.plot(range(1, len(valid_similarity) + 1), valid_similarity, label="Valid similarity")
plt.xlabel("Epoch")
plt.ylabel("Cosine Similarity")
plt.legend()
plt.savefig('./plots/plot.pdf')
'''
