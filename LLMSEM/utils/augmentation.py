import random
import numpy as np
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from collections import Counter
import pandas as pd

class FinancialTextAugmenter:
    """
    Classe pour augmenter les données textuelles financières
    avec plusieurs techniques adaptées au domaine financier
    """
    
    def __init__(self):
        # Synonymes financiers pour préserver le contexte
        self.synonym_aug = naw.SynonymAug(aug_src='wordnet')
        
        # Back translation (nécessite internet)
        # self.back_translation_aug = naw.BackTranslationAug(
        #     from_model_name='facebook/wmt19-en-de',
        #     to_model_name='facebook/wmt19-de-en'
        # )
    
    def synonym_replacement(self, text, n=2):
        """Remplace n mots par des synonymes"""
        try:
            augmented = self.synonym_aug.augment(text, n=1)
            return augmented if isinstance(augmented, str) else augmented[0]
        except:
            return text
    
    def random_insertion(self, text, n=2):
        """Insère des mots aléatoires à des positions aléatoires"""
        words = text.split()
        if len(words) < 3:
            return text
        
        financial_words = ['significant', 'substantial', 'notable', 'considerable', 
                          'major', 'minor', 'slight', 'moderate']
        
        for _ in range(n):
            word_to_insert = random.choice(financial_words)
            position = random.randint(0, len(words))
            words.insert(position, word_to_insert)
        
        return ' '.join(words)
    
    def random_swap(self, text, n=2):
        """Échange aléatoirement des mots dans la phrase"""
        words = text.split()
        if len(words) < 2:
            return text
        
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def random_deletion(self, text, p=0.1):
        """Supprime aléatoirement des mots avec probabilité p"""
        words = text.split()
        if len(words) == 1:
            return text
        
        new_words = [word for word in words if random.random() > p]
        
        if len(new_words) == 0:
            return random.choice(words)
        
        return ' '.join(new_words)
    
    def augment_text(self, text, num_aug=1, techniques=['synonym', 'swap', 'insertion']):
        """
        Applique plusieurs techniques d'augmentation
        
        Args:
            text: texte à augmenter
            num_aug: nombre de variations à générer
            techniques: liste des techniques à utiliser
        """
        augmented_texts = []
        
        for _ in range(num_aug):
            aug_text = text
            technique = random.choice(techniques)
            
            if technique == 'synonym':
                aug_text = self.synonym_replacement(aug_text)
            elif technique == 'swap':
                aug_text = self.random_swap(aug_text)
            elif technique == 'insertion':
                aug_text = self.random_insertion(aug_text)
            elif technique == 'deletion':
                aug_text = self.random_deletion(aug_text)
            
            augmented_texts.append(aug_text)
        
        return augmented_texts


def balance_dataset(df, target_column='Sentiment', text_column='Sentence', 
                   strategy='oversample', augmenter=None):
    """
    Équilibre le dataset en utilisant l'augmentation de données
    
    Args:
        df: DataFrame pandas
        target_column: nom de la colonne cible
        text_column: nom de la colonne texte
        strategy: 'oversample' ou 'balance_to_max'
        augmenter: instance de FinancialTextAugmenter
    
    Returns:
        DataFrame équilibré
    """
    if augmenter is None:
        augmenter = FinancialTextAugmenter()
    
    # Compter les occurrences de chaque classe
    class_counts = df[target_column].value_counts()
    print(f"Distribution originale:\n{class_counts}\n")
    
    if strategy == 'oversample':
        # Sursampler jusqu'à la classe majoritaire
        max_count = class_counts.max()
        target_count = max_count
    elif strategy == 'balance_to_max':
        # Équilibrer à la classe majoritaire
        max_count = class_counts.max()
        target_count = max_count
    else:
        raise ValueError("Strategy doit être 'oversample' ou 'balance_to_max'")
    
    augmented_data = []
    
    for sentiment_class in class_counts.index:
        class_df = df[df[target_column] == sentiment_class]
        current_count = len(class_df)
        
        # Ajouter les données originales
        augmented_data.append(class_df)
        
        # Calculer combien d'exemples augmenter
        needed = target_count - current_count
        
        if needed > 0:
            print(f"Classe {sentiment_class}: augmentation de {needed} exemples")
            
            # Sélectionner aléatoirement des exemples à augmenter
            samples_to_augment = class_df.sample(n=min(needed, len(class_df)), 
                                                  replace=True, random_state=42)
            
            new_texts = []
            new_sentiments = []
            
            for _, row in samples_to_augment.iterrows():
                # Générer 1 ou plusieurs variations
                num_variations = max(1, needed // len(samples_to_augment))
                augmented_texts = augmenter.augment_text(
                    row[text_column], 
                    num_aug=num_variations,
                    techniques=['synonym', 'swap', 'insertion']
                )
                
                new_texts.extend(augmented_texts)
                new_sentiments.extend([sentiment_class] * len(augmented_texts))
            
            # Limiter au nombre nécessaire
            new_texts = new_texts[:needed]
            new_sentiments = new_sentiments[:needed]
            
            # Créer un nouveau DataFrame avec les données augmentées
            aug_df = pd.DataFrame({
                text_column: new_texts,
                target_column: new_sentiments
            })
            
            augmented_data.append(aug_df)
    
    # Combiner toutes les données
    balanced_df = pd.concat(augmented_data, ignore_index=True)
    
    # Mélanger le dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nDistribution après augmentation:\n{balanced_df[target_column].value_counts()}")
    
    return balanced_df


