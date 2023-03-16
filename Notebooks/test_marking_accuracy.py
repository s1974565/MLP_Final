import pandas as pd
import json
import numpy as np

#loading and printing
df = pd.read_pickle("valid.pickle")

with open('valid.json', 'r') as f:
    data = json.load(f)
manual = pd.DataFrame(data)["code"]


generated = df["code"].iloc[:20]


measure = []
for i in range (0, 20):
	manual_mask = manual.loc[i]
	generated_mask = generated.loc[i]
	true_count = manual_mask.count("<mask>")
	if manual_mask == generated_mask:
		measure.append({"i": i, "TP": true_count, "FN": 0, "FP": 0})
	else:
		manual_array = np.array(manual_mask.split())
		generated_array = np.array(generated_mask.split())
		arr = np.where(manual_array != generated_array)[0]
		different_len = len(arr)
		false_neg = (" ".join(manual_array[arr])).count("<mask>")
		true_pos = true_count - false_neg
		false_pos = different_len - false_neg
		measure.append({"i": i, "TP": true_pos, "FN": false_neg, "FP": false_pos})

TP_sum = 0
FP_sum = 0
FN_sum = 0

for dict in measure:
	TP_sum += dict["TP"]
	FP_sum += dict["FP"]
	FN_sum += dict["FN"]

precision = TP_sum/(TP_sum + FP_sum)
recall = TP_sum/(TP_sum + FN_sum)
f_measure = (2*precision*recall)/(precision+recall)

print("Precision: %.2f" % precision)
print("recall: %.2f" % recall)
print("F-measure: %.2f" % f_measure)