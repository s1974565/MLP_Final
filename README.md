RoBERTvar: Improving code variable name suggestions with fine-tuned language models

To access the model from huggingface using a jupyter notebook use the following (you currently require an access token from gnathoi):

from huggingface_hub import notebook_login

notebook_login()

model = RobertaForMaskedLM.from_pretrained('gnathoi/roBERTvar5pc', use_auth_token=True)

tokenizer = RobertaTokenizer.from_pretrained('gnathoi/roBERTvar5pc', use_auth_token=True)
