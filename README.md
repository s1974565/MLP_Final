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
