import os
os.environ["WANDB_DISABLED"] = "true"

from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForCausalLM, TrainingArguments, \
	Trainer, DataCollatorWithPadding
import datasets
import torch
from utils.loader import DataLoader


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


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


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

	# tokenized_dec = dataset_dec.map(tokenize_data(tokenizer=), batched=True)
	tokenizer.pad_token = tokenizer.eos_token
	tokenized_val = dataset_truth_val.map(tokenize_data, batched=True, remove_columns=['text'])
	tokenized_train = dataset_truth_train.map(tokenize_data, batched=True, remove_columns=['text'])

	# Collate and chunk datasets
	# lm_train = tokenized_train.map(
    # group_texts,
    # batched=True,
    # batch_size=1000,
    # num_proc=4,
	# )
	# lm_val = tokenized_val.map(
    # group_texts,
    # batched=True,
    # batch_size=1000,
    # num_proc=4,
	# )
	lm_train = tokenized_train
	lm_val = tokenized_val

	del dataset_truth_val
	del dataset_truth_train
	del truth_data_val
	del truth_data_train
	del truth_data

	# Set padding token and set mlm false to use the inputs as the labels shifted to right by one
	data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
	collator = DataCollatorWithPadding(tokenizer, padding='max_length', max_length=block_size)

	# Load model
	model = AutoModelForCausalLM.from_pretrained("distilgpt2")
	training_args = TrainingArguments(
		output_dir="checkpoints/distilgpt2",
		evaluation_strategy="epoch",
		learning_rate=2e-5,
		num_train_epochs=4,
		weight_decay=0.01,
		per_device_train_batch_size=10,
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

	import math
	eval_results = trainer.evaluate()
	print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
