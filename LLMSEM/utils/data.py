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

train_df = ds["train"].map(clean_text)
validation_df = ds["validation"].map(clean_text)

# conversion en Pandas pour CSV
train_df.to_pandas().to_csv(r"LLMSEM\data\train_df.csv", index=False)
validation_df.to_pandas().to_csv(r"LLMSEM\data\validation_df.csv", index=False)


# distribution des sentiments
train_df.to_pandas().groupby("label").size().plot(kind="bar")
validation_df.to_pandas().groupby("label").size().plot(kind="bar")
plt.savefig(r"LLMSEM\results\distribution_sentiments.png")
plt.show()