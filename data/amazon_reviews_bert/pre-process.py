#!/usr/bin/env python3

from cleantext import clean
from urllib.parse import unquote

import pandas as pd

SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                    "eos_token": "<|EOS|>",
                    "unk_token": "<|UNK|>",                    
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}

def load_data(path):
    return pd.read_csv(path)


def transform(df):
    def decode(reviews):
        return unquote(reviews)

    def clean_text(reviews):
        return clean(reviews, lowercase=False, extra_spaces=True, stopwords=False, stemming=False, numbers=False, punct=False)

    def sub_line_breaks(reviews):
        return reviews.replace("<br />", SPECIAL_TOKENS["sep_token"])

    def strip_emojis(df):
        return df.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
    
    df["REVIEW_TEXT"] = df["REVIEW_TEXT"].apply(decode)    
    df["REVIEW_TEXT"] = df["REVIEW_TEXT"].apply(clean_text)
    df["REVIEW_TEXT"] = df ["REVIEW_TEXT"].apply(sub_line_breaks)

    df["REVIEW_TITLE"] = df["REVIEW_TITLE"].apply(decode)
    df["REVIEW_TITLE"] = df["REVIEW_TITLE"].apply(clean_text)
    df["REVIEW_TITLE"] = df["REVIEW_TITLE"].apply(sub_line_breaks)
    strip_emojis(df)
    return df

def write_data(df, path):
    df.to_csv(path, index=False)

if __name__ == "__main__":
    TEST_PATH = "./test/amazon_reviews.txt"
    TRAIN_PATH = "./train/amazon_reviews.txt"
    
    df = load_data(TEST_PATH)
    write_data(transform(df), "./test.txt")
    
