import os
os.environ["WANDB_DISABLED"] = "true"

from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding, AutoConfig
from torch.utils.data import Dataset
import datasets
import torch
from utils.loader import DataLoader
from utils.configs import config
import pandas as pd
import math

SPECIAL_TOKENS = config['special_tokens']
VAL_RATIO = 0.2
BLOCK_SIZE = 1024
tokenizer = AutoTokenizer.from_pretrained('distilgpt2', use_fast=True)


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        text, category = [], []
        for k, v in data[['PRODUCT_CATEGORY', 'REVIEW_TEXT']].iterrows():
            text.append(v[1])
            category.append(v[2])

        self.tokenizer = tokenizer
        self.text = text
        self.category = category  

    def __len__(self):
        return len(self.text)

    def __getitem__(self, i):
        input = SPECIAL_TOKENS['bos_token'] + self.category[i] + SPECIAL_TOKENS['sep_token'] + \
                self.text[i] + SPECIAL_TOKENS['eos_token']

        encodings_dict = tokenizer(input,                                   
                                   truncation=True, 
                                   max_length=1024, 
                                   padding="max_length")   
        
        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']
        
        return {'label': torch.tensor(input_ids),
                'input_ids': torch.tensor(input_ids), 
                'attention_mask': torch.tensor(attention_mask)}


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
	truth_data_val = truth_data.iloc[:int(VAL_RATIO * len(truth_data))]
	truth_data_train = truth_data.iloc[int(VAL_RATIO * len(truth_data)):]

	lm_train = CustomDataset(truth_data_train, tokenizer)
	lm_val = CustomDataset(truth_data_val, tokenizer)

	del truth_data_val
	del truth_data_train
	del truth_data

	# Set padding token and set mlm false to use the inputs as the labels shifted to right by one
	collator = DataCollatorWithPadding(tokenizer, padding='max_length', max_length=block_size)

	# Load model
	config = AutoConfig.from_pretrained('distilgpt2', 
												bos_token_id=tokenizer.bos_token_id,
												eos_token_id=tokenizer.eos_token_id,
												sep_token_id=tokenizer.sep_token_id,
												pad_token_id=tokenizer.pad_token_id,
												output_hidden_states=False)
	model = AutoModelForCausalLM.from_pretrained("distilgpt2", config=config)
	model.resize_token_embeddings(len(tokenizer)) # Resize model to fit special tokens
	
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
		train_dataset=lm_train,
		eval_dataset=lm_val,
		data_collator = collator,
		tokenizer=tokenizer
	)

	torch.cuda.empty_cache()

	trainer.train()

	eval_results = trainer.evaluate()
	print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
