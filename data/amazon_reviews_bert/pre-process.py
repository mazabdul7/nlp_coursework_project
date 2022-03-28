#!/usr/bin/env python3

from cleantext import clean
from urllib.parse import unquote

import re
import pandas as pd

SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                    "eos_token": "<|EOS|>",
                    "unk_token": "<|UNK|>",                    
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}


HTML_ENCODINGS = { "&#124;": "|",
                   "&#34;": '""',
                   "&#62;": ">",
                   "&#60;": "<",
                   "&#8482": SPECIAL_TOKENS["unk_token"]}

def load_data(path):
    return pd.read_csv(path)


def transform(df):
    def decode(reviews):
        return unquote(reviews)

    def clean_text(reviews):
        return clean(reviews, lowercase=False, extra_spaces=True, stopwords=False, stemming=False, numbers=False, punct=False, clean_all=False)

    def sub_line_breaks_and_special_chars(reviews):
        for key, value in HTML_ENCODINGS.items():
            reviews = reviews.replace(key, value)
        return reviews.replace("<br />", SPECIAL_TOKENS["sep_token"])

    def strip_emojis(reviews):
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
        return emoji_pattern.sub(SPECIAL_TOKENS["unk_token"], reviews)
    
    df["REVIEW_TEXT"] = df["REVIEW_TEXT"].apply(decode)    
    df["REVIEW_TEXT"] = df["REVIEW_TEXT"].apply(clean_text)
    df["REVIEW_TEXT"] = df ["REVIEW_TEXT"].apply(strip_emojis)
    df["REVIEW_TEXT"] = df ["REVIEW_TEXT"].apply(sub_line_breaks_and_special_chars)
    df["REVIEW_TITLE"] = df["REVIEW_TITLE"].apply(decode)
    df["REVIEW_TITLE"] = df["REVIEW_TITLE"].apply(clean_text)
    df["REVIEW_TITLE"] = df ["REVIEW_TITLE"].apply(strip_emojis)
    df["REVIEW_TITLE"] = df["REVIEW_TITLE"].apply(sub_line_breaks_and_special_chars)
    
    return df

def write_data(df, path):
    df.to_csv(path, index=False)

if __name__ == "__main__":
    TEST_PATH = "./test/amazon_reviews.txt"
    TRAIN_PATH = "./train/amazon_reviews.txt"
    
    df = load_data(TEST_PATH)
    write_data(transform(df), "./test.txt")
    
