from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForCausalLM, TrainingArguments, Trainer
import datasets
import pandas as pd
import torch


def load_amazon(deceptive=False, all=False):
    """ Loads data from the Amazon dataset. Label 1 (1) is deceptive reviews, Label 2 (0) is authentic.

    Args:
        deceptive (bool, optional): Return deceptive instead of authentic.

    Returns:
        Dataframe: Returns a dataframe of the reviews.
    """
    # TODO: os path
    data = pd.read_table('./data/amazon_reviews/amazon_reviews.txt')
    data.loc[data['LABEL'] == '__label2__', 'LABEL'] = 1
    data.loc[data['LABEL'] == '__label1__', 'LABEL'] = 0
    
    if all: 
        return data
    else:
        if deceptive:
            return data.get(data['LABEL'] == 1)
        return data.get(data['LABEL'] == 0)

def df_to_dataset_obj(dataframe, columns):
    dataset = datasets.Dataset.from_pandas(dataframe[columns])
    dataset = dataset.remove_columns('__index_level_0__')
    dataset = dataset.rename_column('LABEL', 'labels')
    dataset = dataset.rename_column('REVIEW_TEXT', 'text')
    
    return dataset

def tokenize_data(inputs):
    return tokenizer(inputs['text'], padding='max_length', truncation=True)



val_split = 0.2

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

# Load datasets
#dec_data = load_amazon(deceptive=True)
truth_data = load_amazon()
truth_data = truth_data.sample(frac=1)
truth_data_val = truth_data.iloc[:int(0.2*len(truth_data))]
truth_data_train = truth_data.iloc[int(0.2*len(truth_data)):]


# Clean and convert to Dataset objects
#dataset_dec = df_to_dataset_obj(dec_data, ['LABEL', 'REVIEW_TEXT'])
dataset_truth_val = df_to_dataset_obj(truth_data_val, ['LABEL', 'REVIEW_TEXT'])
dataset_truth_train = df_to_dataset_obj(truth_data_train, ['LABEL', 'REVIEW_TEXT'])


tokenizer.pad_token = tokenizer.eos_token
#tokenized_dec = dataset_dec.map(tokenize_data(tokenizer=), batched=True)
tokenized_val = dataset_truth_val.map(tokenize_data, batched=True, remove_columns=['labels', 'text'])
tokenized_train = dataset_truth_train.map(tokenize_data, batched=True, remove_columns=['labels', 'text'])

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

torch.cuda.empty_cache()

trainer.train()
