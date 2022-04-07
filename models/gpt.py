from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

MODEL = 'distilgpt2'

class GPT2:
    ''' Wrapper class for managing and loading GPT2 models. '''
    def __init__(self, model_path=None, full_model=False, special_tokens=None) -> None:
        self.tokenizer = self.get_tokenizer(special_tokens)
        self.model = self.get_model(self.tokenizer, special_tokens=special_tokens, load_model_path=model_path, full_model=full_model)
        
    def get_tokenizer(self, special_tokens=None):
        tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

        if special_tokens:
            tokenizer.add_special_tokens(special_tokens)
        return tokenizer
    
    def get_model(self, tokenizer, special_tokens=None, load_model_path=None, full_model=False):
        if full_model:
            model = AutoModelForCausalLM.from_pretrained(load_model_path)
            model.cuda()
            return model 
        
        if special_tokens:
            config = AutoConfig.from_pretrained(MODEL, 
                                                bos_token_id=tokenizer.bos_token_id,
                                                eos_token_id=tokenizer.eos_token_id,
                                                sep_token_id=tokenizer.sep_token_id,
                                                pad_token_id=tokenizer.pad_token_id,
                                                output_hidden_states=False)
        else: 
            config = AutoConfig.from_pretrained(MODEL,                                     
                                                pad_token_id=tokenizer.eos_token_id,
                                                output_hidden_states=False)    

        model = AutoModelForCausalLM.from_pretrained(MODEL, config=config)

        if special_tokens:
            model.resize_token_embeddings(len(tokenizer))

        if load_model_path:
            model.load_state_dict(torch.load(load_model_path))#map_location=torch.device('cpu')))

        model.to(torch.device('cuda:0'))
        return model
    
    def generate_text(self, prompt, category, print_output=True, **kwargs):
        generated_outputs = []
        
        # Tokenize prompt
        tokenized_prompt = self.tokenizer.encode(prompt, return_tensors='pt').to('cuda:0')
        
        # Language modelling
        output = self.model.generate(tokenized_prompt, **kwargs)
        
        for i, o in enumerate(output):
            gen_txt = self.tokenizer.decode(o, skip_special_tokens=True)
            gen_txt = gen_txt[len(category):]
            truncated_txt = gen_txt.split('.')
            truncated_txt = '.'.join(truncated_txt[:-1]) + '.'
            generated_outputs.append(truncated_txt)
            
            if print_output:
                print(truncated_txt + '\n')
                
        return generated_outputs