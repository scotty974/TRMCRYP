import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from utils.augmentation import FinancialTextAugmenter, balance_dataset
df = pd.read_csv(r"LLMSEM\data\data.csv")

# clean data 
df = df.dropna()
df = df.reset_index(drop=True)


# transform label to int
# 0 = positive, 1 = neutral, 2 = negative
label_encoder = LabelEncoder()
df['Sentiment'] = label_encoder.fit_transform(df['Sentiment'])
augmenter = FinancialTextAugmenter()
count_label = df['Sentiment'].value_counts()

plt.bar(count_label.index, count_label.values)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Distribution of Sentiment')
plt.savefig(r"LLMSEM\results\data\distribution_sentiment.png")
# plt.show()

balanced_df = balance_dataset(
        df, 
        target_column='Sentiment', 
        text_column='Sentence',
        strategy='oversample',
        augmenter=augmenter
    )
    
# Sauvegarder le dataset équilibré
balanced_df.to_csv(r"LLMSEM\data\data_balanced.csv", index=False)
    
print(f"\nTaille originale: {len(df)}")
print(f"Taille après augmentation: {len(balanced_df)}")

plt.bar(balanced_df['Sentiment'].value_counts().index, balanced_df['Sentiment'].value_counts().values)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Distribution of Sentiment')
plt.savefig(r"LLMSEM\results\data\distribution_sentiment_balanced.png")
# plt.show()
