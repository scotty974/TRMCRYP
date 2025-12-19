from datasets import load_dataset
import re
import matplotlib.pyplot as plt


ds = load_dataset("zeroshot/twitter-financial-news-sentiment")

def clean_text(example):
    text = example["text"]
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^A-Za-zÀ-ÖØ-öø-ÿ\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    example["text"] = text.strip().lower()
    return example

train_df = ds["train"].map(clean_text).to_pandas()
validation_df = ds["validation"].map(clean_text).to_pandas()

# Distribution des sentiments
train_df.groupby("label").size().plot(kind="bar")
plt.savefig(r"LLMSEM\results\train_distribution.png")
plt.show()

validation_df.groupby("label").size().plot(kind="bar")
plt.savefig(r"LLMSEM\results\val_distribution.png")
plt.show()
