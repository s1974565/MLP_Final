# RoBERTvar: Improving code variable name suggestions with fine-tuned language models

[Find the model(s) here.](https://huggingface.co/gnathoi)

To access the model from huggingface using a jupyter notebook use the following:



```python
from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline

auth_token = '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' # this token is read only

model = RobertaForMaskedLM.from_pretrained('gnathoi/RoBERTvar', use_auth_token=auth_token)
tokenizer = RobertaTokenizer.from_pretrained('gnathoi/RoBERTvar', use_auth_token=auth_token)
var_gen = pipeline('fill-mask', model=model, tokenizer=tokenizer) 
```

Generally to run the scripts and notebooks in this repo you will need to install some additional packages into your conda environment. Assuming you have the standard python packages, numpy, matplotlib, etc.. you will need pytorch + cuda. To install the remaining additional packages the following pip command in the terminal is likely all you need to do.

```bash
pip install -U sentence-transformers
```
