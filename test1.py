import torch

from datasets import load_dataset
# from transformers import AutoModelForSeq2SeqLM
# from transformers import AutoTokenizer
from transformers import GenerationConfig

print("It worked!")

# huggingface_dataset_name = "knkarthick/dialogsum"
# # dataset = load_dataset(huggingface_dataset_name)
# dataset = load_dataset("glue", "mrpc", split="train")
# print("Dataset Loaded!")

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl").to(device)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

inputs = tokenizer("Answer the following question by reasoning step-by-step: I had rice with fish and later had milk tea. Why am I having headache and neck pain?", return_tensors="pt").to(device)
# model = model.to(device)
outputs = model.generate(**inputs, max_new_tokens=200).to(device)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
# inputs = tokenizer("If I didn't. Then what is the cause of headache and neck pain, explain in detail?", return_tensors="pt")
# outputs = model.generate(**inputs, max_new_tokens=200)
# print(tokenizer.batch_decode(outputs, skip_special_tokens=True))