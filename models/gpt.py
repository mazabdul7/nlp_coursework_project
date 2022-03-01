import torch
import numpy as np
import pandas as pd
from transformers import pipeline

class GPT:
    def __init__(self) -> None:
        self.generator = self.instantiate_gpt('text-generation', 'gpt2')
        
    def instantiate_gpt(self, task, model):
        return pipeline(task, model=model)
    
    def generate_text(self, pretext, max_length, num_sequences):
        generated_txt = self.generator(pretext, max_length=max_length, num_return_sequences=num_sequences)
        outputs = [list(txt.values())[0] for txt in generated_txt]
        
        return outputs