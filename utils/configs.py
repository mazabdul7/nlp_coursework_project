config = {
	'base_path': r'data',
	'gold_path': r'op_spam_v1.4',
	'dec_path': r'dec',  # Both polarities have deceptive reviews from MTurk
	'truth_path': r'truth',
	# The negative polarity has authentic web reviews, positive polarity has authentic tripadvisor reviews
	'pos_path': r'positive_polarity',
	'neg_path': r'negative_polarity',
	'amazon_path': r'amazon_reviews',
	'special_tokens': { "bos_token": "<|BOS|>",
                    "eos_token": "<|EOS|>",
                    "unk_token": "<|UNK|>",                    
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}
}
