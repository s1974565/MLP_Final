# RoBERTvar: Improving code variable name suggestions with fine-tuned language models

[Find the model(s) here.](https://huggingface.co/gnathoi)

To access the model from huggingface using a jupyter notebook use the following (you currently require an access token from gnathoi):

```python
from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline
from huggingface_hub import notebook_login
notebook_login()

model = RobertaForMaskedLM.from_pretrained('gnathoi/modelname', use_auth_token=True)
tokenizer = RobertaTokenizer.from_pretrained('gnathoi/modelname', use_auth_token=True)
```

Generally to run the scripts and notebooks in this repo you will need to install some additional packages into your conda environment. Assuming you have the standard python packages, numpy, matplotlib, etc.. you will need pytorch + cuda. To install the remaining additional packages the following pip command in the terminal is likely all you need to do.

```bash
pip install -U sentence-transformers
```
