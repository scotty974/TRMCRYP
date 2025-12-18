import pandas as pd
import re

splits = {'train': 'split/train-00000-of-00001.parquet', 'validation': 'split/validation-00000-of-00001.parquet', 'test': 'split/test-00000-of-00001.parquet'}
train_df = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["train"])
validation_df = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["validation"])
test_df = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["test"])

def clean_text(text):
    text = re.sub(r'https?://\S+', '', text)  # supprime les URLs
    text = re.sub(r'#\w+', '', text)          # supprime les hashtags
    text = re.sub(r'@\w+', '', text)          # supprime les mentions
    text = re.sub(r'[^A-Za-zÀ-ÖØ-öø-ÿ\s]', ' ', text)  # supprime les caractères spéciaux
    text = re.sub(r'\s+', ' ', text)          # remplace les multiples espaces par un seul
    return text.strip().lower()                # enlève les espaces en début/fin et met en minuscule

train_df['text'] = train_df['text'].apply(clean_text)
validation_df['text'] = validation_df['text'].apply(clean_text)
test_df['text'] = test_df['text'].apply(clean_text)

print("Data Cleaned")

label_mapping = {}  # dictionnaire pour associer label → texte exemple
for label in sorted(train_df['label'].unique()):
    example_text = train_df[train_df['label'] == label]['text'].iloc[0]
    print(f"\nLabel {label}: {example_text}")
    label_mapping[label] = example_text

# Afficher le mapping final
print("\nMapping des labels avec exemple de texte:")
for label, text in label_mapping.items():
    print(f"{label} → {text}")

for label in sorted(train_df['label'].unique()):
    print(f"\nLabel {label}:")
    texts = train_df[train_df['label'] == label]['text'].head(5)  # afficher 5 exemples
    for t in texts:
        print("-", t)

