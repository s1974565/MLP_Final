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

model_path = './saved_models/model_0/'
model_name = 'model_0'

model_b1 = RobertaForMaskedLM.from_pretrained(model_path)
tokenizer_b1 = RobertaTokenizer.from_pretrained(model_path)

hub_path = 'gnathoi/RoBERTvar' # change model name 
auth_token = ''  # make sure to add woken

model_b1.push_to_hub(hub_path, use_auth_token=auth_token)
tokenizer_b1.push_to_hub(hub_path, use_auth_token=auth_token)
