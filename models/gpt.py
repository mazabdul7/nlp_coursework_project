from transformers import pipeline, AutoTokenizer, AutoModelWithLMHead, AdamW, get_linear_schedule_with_warmup

class GPT2:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2') # Tokenizer
        self.model = AutoModelWithLMHead.from_pretrained('gpt2-large') # Weights
        self.generator = self.instantiate_gen('text-generation', self.model, self.tokenizer, 'pt')
        
    def instantiate_gen(self, task, model, tokenizer, framework):
        return pipeline(task, model=model, tokenizer=tokenizer, framework=framework)
    
    def set_optimiser(self, epochs, lr=1e-4, warmup=0.1):
        self.optimizer = AdamW(self.generator.parameters(), lr=lr, t_total=epochs, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup*epochs, num_training_steps=epochs) # Possibly use a warmup scheduler
        
        return self.optimizer, scheduler
    
    def generate_text(self, pretext, num_sequences, max_length=1000):
        generated_txt = self.generator(pretext, max_length=max_length, num_return_sequences=num_sequences)
        outputs = [list(txt.values())[0] for txt in generated_txt]
        
        return outputs