import os
os.environ["WANDB_DISABLED"] = "true"

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, \
	Trainer, DataCollatorWithPadding
import datasets
import torch
from utils.loader import DataLoader
import math


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilgpt2', use_fast=True)
val_ratio = 0.2
block_size = 1024


def df_to_dataset_obj(dataframe, columns):
	dataset = datasets.Dataset.from_pandas(dataframe[columns])
	dataset = dataset.remove_columns('__index_level_0__')
	dataset = dataset.rename_column('LABEL', 'labels')
	dataset = dataset.rename_column('REVIEW_TEXT', 'text')

	return dataset


def tokenize_data(inputs):
	tokens = tokenizer(inputs['text'], padding='max_length', truncation=True, max_length=block_size)
	tokens['labels'] = tokens['input_ids'].copy()
	return tokens


if __name__ == '__main__':
	# Load datasets
	data_loader = DataLoader()
	truth_data = data_loader.load_amazon(deceptive=False)
	truth_data = truth_data.sample(frac=1)
	truth_data_val = truth_data.iloc[:int(val_ratio * len(truth_data))]
	truth_data_train = truth_data.iloc[int(val_ratio * len(truth_data)):]

	# Clean and convert to Dataset objects
	# dataset_dec = df_to_dataset_obj(dec_data, ['LABEL', 'REVIEW_TEXT'])
	dataset_truth_val = df_to_dataset_obj(truth_data_val, ['LABEL', 'REVIEW_TEXT'])
	dataset_truth_train = df_to_dataset_obj(truth_data_train, ['LABEL', 'REVIEW_TEXT'])

	# Tokenize data
	tokenizer.pad_token = tokenizer.eos_token
	tokenized_val = dataset_truth_val.map(tokenize_data, batched=True, remove_columns=['text'])
	tokenized_train = dataset_truth_train.map(tokenize_data, batched=True, remove_columns=['text'])

	lm_train = tokenized_train
	lm_val = tokenized_val

	del dataset_truth_val
	del dataset_truth_train
	del truth_data_val
	del truth_data_train
	del truth_data

	# Set padding token and set mlm false to use the inputs as the labels shifted to right by one
	# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
	collator = DataCollatorWithPadding(tokenizer, padding='max_length', max_length=block_size)

	# Load model
	model = AutoModelForCausalLM.from_pretrained("distilgpt2")
	
	### UNCOMMENT IF FREEZING LAYERS ###
	# for parameter in model.parameters(): # Freeze all layers
	#     parameter.requires_grad = False

	# for i, m in enumerate(model.transformer.h):   
	#     #Only un-freeze the last n transformer blocks
	#     if i >= 6:
	#         for parameter in m.parameters():
	#             parameter.requires_grad = True 
	# for parameter in model.transformer.ln_f.parameters():        
	#     parameter.requires_grad = True

	# for parameter in model.lm_head.parameters():        
	#     parameter.requires_grad = True
 
	training_args = TrainingArguments(
		output_dir="checkpoints/distilgpt2",
		evaluation_strategy="epoch",
		learning_rate=2e-5,
		num_train_epochs=6,
		weight_decay=0.01,
		per_device_train_batch_size=4,
		save_strategy='epoch'
	)
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=tokenized_train,
		eval_dataset=tokenized_val,
		data_collator = collator,
		tokenizer=tokenizer
	)

	torch.cuda.empty_cache()

	trainer.train()

	eval_results = trainer.evaluate()
	print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
