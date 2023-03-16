import pandas as pd
import glob
import re

def jsonl_list_to_dataframe(file_list, columns=None):
    """Load a list of jsonl.gz files into a pandas DataFrame."""
    return pd.concat([pd.read_json(f,
                                   orient='records', 
                                   compression='gzip',
                                   lines=True)[columns] 
                      for f in file_list], sort=False)

def get_dfs(path):
	"""Grabs the different data splits and converts them into dataframes"""
	dfs = []
	for split in ["train", "valid", "test"]:
		files = sorted(glob.glob(path+"/"+split+"**/*.gz"))
		df = jsonl_list_to_dataframe(files, ["code", "repo"])
		dfs.append(df)
	return dfs

def mask_variable_names(code, mask_prob=0.5):
	"""
	Mask the values of variables in a code with a certain probability.
	"""
	masked_code = code
	# Regular expression pattern to match variable assignments
	pattern = r"\b([a-zA-Z_]\w*(?:\s*[,=]\s*[a-zA-Z_]\w*)*?\s=\s[^#\n]*)"
	for match in re.findall(pattern, code):
		# Check if the value should be masked
		# Replace the value with "MASKED"
		variables, _ = match.rsplit("=", 1)
		if "(" in variables or ")" in variables:
			continue
		#print(variables)
		variables = list(set(re.split(",|=", variables)))
		masked_var = match
		for var in variables:
			temp = var
			var = var.replace(var.strip(), "<mask>")
			masked_var = masked_var.replace(temp, var)
		masked_code = masked_code.replace(match, masked_var)
	return masked_code

df_trn, df_val, df_tst = get_dfs("python/python/final/jsonl")

df_tst["code"] = df_tst["code"].apply(lambda code: mask_variable_names(code))
df_val["code"] = df_val["code"].apply(lambda code: mask_variable_names(code))

df_trn.to_pickle("train.pickle")
df_val.to_pickle("valid.pickle")
df_tst.to_pickle("test.pickle")


