from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForCausalLM, TrainingArguments, \
	Trainer
from utils.loader import DataLoader
import datasets
import pandas as pd

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
val_ratio = 0.2


def list_to_dataset_obj(data):
	data_df = pd.DataFrame(data, columns=['text'])
	dataset = datasets.Dataset.from_pandas(data_df)

	return dataset


def tokenize_data(inputs):
	return tokenizer(inputs['text'], padding='max_length', truncation=True)


if __name__ == '__main__':
	# Load datasets
	loader = DataLoader()
	truth_data = loader.load_gold_data('truth')
	truth_data_val = truth_data[:int(val_ratio * len(truth_data))]
	truth_data_train = truth_data[int(val_ratio * len(truth_data)):]

	# Clean and convert to Dataset objects
	dataset_truth_val = list_to_dataset_obj(truth_data_val)
	dataset_truth_train = list_to_dataset_obj(truth_data_train)

	tokenizer.pad_token = tokenizer.eos_token
	tokenized_val = dataset_truth_val.map(tokenize_data, batched=True, remove_columns=['text'])
	tokenized_train = dataset_truth_train.map(tokenize_data, batched=True, remove_columns=['text'])

	del dataset_truth_val
	del dataset_truth_train
	del truth_data_val
	del truth_data_train
	del truth_data

	# Set padding token and use mlm to use the inputs as the labels shifted to right by one
	tokenizer.pad_token = tokenizer.eos_token
	data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

	# Load model
	model = AutoModelForCausalLM.from_pretrained("distilgpt2")
	training_args = TrainingArguments(
		output_dir="sample_data",
		evaluation_strategy="epoch",
		learning_rate=2e-5,
		num_train_epochs=5.0,
		weight_decay=0.01,
		per_device_train_batch_size=4,
		save_strategy='epoch'

	)
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=tokenized_train,
		eval_dataset=tokenized_val,
		data_collator=data_collator
	)

	trainer.train()
