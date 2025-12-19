
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention
from tensorflow.keras import Model, Sequential
import json
from datetime import datetime
from typing import Dict, Tuple, List


CONFIG = {
    'vocab_size': 5000,
    'seq_len': 128,
    'embed_dim': 256,
    'batch_size': 32,
    'epochs': 10,
    'attention_heads': 4,
    'dropout_rate': 0.1,
    'learning_rate': 1e-4
}

class SentimentAnalyzer(Model):
    """
    Analyse le sentiment des nouvelles financières
    Architecture optimisée pour le NLP financier
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Architecture principale
        self.embedding = tf.keras.layers.Embedding(
            config['vocab_size'],
            config['embed_dim']
        )
        
        self.attention = MultiHeadAttention(
            num_heads=config['attention_heads'],
            key_dim=config['embed_dim'] // config['attention_heads']
        )
        
        self.layer_norm_1 = LayerNormalization()
        self.layer_norm_2 = LayerNormalization()
        
        # Réseau feed-forward optimisé
        self.ffn = Sequential([
            Dense(config['embed_dim'] * 2, activation='gelu'),
            tf.keras.layers.Dropout(config['dropout_rate']),
            Dense(config['embed_dim'])
        ])
        
        # Tête de classification
        self.classifier_head = Sequential([
            Dense(64, activation='gelu'),
            tf.keras.layers.Dropout(config['dropout_rate']),
            Dense(3, activation='softmax')  # bullish/neutral/bearish
        ])
        
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass optimisé"""
        # Embedding
        x = self.embedding(inputs)
        
        # Self-attention
        attn_output = self.attention(x, x)
        x = self.layer_norm_1(x + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(x, training=training)
        x = self.layer_norm_2(x + ffn_output)
        
        # Pooling global
        x = tf.reduce_mean(x, axis=1)
        
        # Classification
        return self.classifier_head(x)


class PricePredictor(Model):
    """
    Prédiction des prix basée sur les séries temporelles
    Combine CNN pour motifs locaux et attention pour dépendances longues
    """
    
    def __init__(self, lookback: int = 30, forecast_horizon: int = 5):
        super().__init__()
        
        # Extraction de caractéristiques temporelles
        self.feature_extractor = Sequential([
            tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
            tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
            tf.keras.layers.GlobalAveragePooling1D()
        ])
        
        # Attention temporelle
        self.temporal_attention = MultiHeadAttention(
            num_heads=4,
            key_dim=32
        )
        
        # Mémoire à long terme
        self.lstm = tf.keras.layers.LSTM(
            128,
            return_sequences=True
        )
        
        # Module de prédiction
        self.prediction_module = Sequential([
            Dense(64, activation='gelu'),
            tf.keras.layers.Dropout(0.1),
            Dense(32, activation='gelu'),
            Dense(forecast_horizon)
        ])
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Prédiction de séries temporelles"""
        # Extraction de caractéristiques
        features = self.feature_extractor(inputs)
        
        # Attention temporelle
        attn_features = self.temporal_attention(features, features)
        
        # Combinaison LSTM
        combined = self.lstm(attn_features[:, None, :])
        
        # Prédiction
        return self.prediction_module(tf.reduce_mean(combined, axis=1))


class InvestmentRecommender(Model):
    """
    Système de recommandation basé sur multiples signaux
    Combine analyse technique, fondamentale et sentiment
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Branches spécialisées
        self.technical_branch = Sequential([
            Dense(32, activation='gelu'),
            LayerNormalization(),
            Dense(16, activation='gelu')
        ])
        
        self.fundamental_branch = Sequential([
            Dense(32, activation='gelu'),
            LayerNormalization(),
            Dense(16, activation='gelu')
        ])
        
        self.sentiment_branch = Sequential([
            Dense(32, activation='gelu'),
            LayerNormalization(),
            Dense(16, activation='gelu')
        ])
        
        # Module de fusion
        self.fusion_module = Sequential([
            Dense(64, activation='gelu'),
            tf.keras.layers.Dropout(config['dropout_rate']),
            LayerNormalization(),
            Dense(32, activation='gelu')
        ])
        
        # Tête de décision
        self.decision_head = Sequential([
            Dense(16, activation='gelu'),
            Dense(3, activation='softmax')  # buy/hold/sell
        ])
        
    def call(self, technical: tf.Tensor, 
             fundamental: tf.Tensor,
             sentiment: tf.Tensor) -> tf.Tensor:
        """Fusion et décision"""
        # Traitement par branches spécialisées
        tech_features = self.technical_branch(technical)
        fund_features = self.fundamental_branch(fundamental)
        sent_features = self.sentiment_branch(sentiment)
        
        # Fusion
        combined = tf.concat([tech_features, fund_features, sent_features], axis=-1)
        fused = self.fusion_module(combined)
        
        # Décision
        return self.decision_head(fused)

class MarketAnalysisOrchestrator:
    """
    Orchestrateur coordonnant les modèles spécialisés
    Gestion de flux de données et décision finale
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialisation des modèles spécialisés
        self.sentiment_model = SentimentAnalyzer(config)
        self.price_model = PricePredictor()
        self.recommender_model = InvestmentRecommender(config)
        
        # Configuration des poids de décision
        self.decision_weights = {
            'sentiment': 0.25,
            'price': 0.35,
            'fundamental': 0.40
        }
        
        # Système de logging
        self.analysis_log = []
        
        print("=" * 60)
        print("SYSTÈME D'ANALYSE BOURSIÈRE MULTI-MODÈLES")
        print("=" * 60)
        print("Modèles initialisés:")
        print("  1. Analyseur de Sentiment")
        print("  2. Prédicteur de Prix")
        print("  3. Recommandateur d'Investissement")
        print("=" * 60)
    
    def analyze_market(self,
                      text_data: np.ndarray,
                      price_data: np.ndarray,
                      technical_data: np.ndarray,
                      fundamental_data: np.ndarray) -> Dict:
        """
        Analyse complète du marché
        
        Args:
            text_data: Données textuelles (nouvelles)
            price_data: Séries temporelles de prix
            technical_data: Indicateurs techniques
            fundamental_data: Données fondamentales
            
        Returns:
            Dict: Résultats de l'analyse
        """
        # Conversion en tensors
        text_tensor = tf.convert_to_tensor(text_data, dtype=tf.int32)
        price_tensor = tf.convert_to_tensor(price_data, dtype=tf.float32)
        tech_tensor = tf.convert_to_tensor(technical_data, dtype=tf.float32)
        fund_tensor = tf.convert_to_tensor(fundamental_data, dtype=tf.float32)
        
        # Exécution parallèle des modèles
        sentiment_result = self.sentiment_model(text_tensor)
        price_prediction = self.price_model(price_tensor)
        recommendation = self.recommender_model(tech_tensor, fund_tensor, sentiment_result)
        
        # Calcul des scores de confiance
        sentiment_confidence = tf.reduce_max(sentiment_result).numpy()
        recommendation_confidence = tf.reduce_max(recommendation).numpy()
        
        # Décision finale pondérée
        final_decision = self._compute_weighted_decision(
            sentiment_result, price_prediction, recommendation
        )
        
        # Préparation des résultats
        analysis_result = {
            'timestamp': datetime.now().isoformat(),
            'sentiment': self._interpret_sentiment(sentiment_result),
            'price_trend': self._analyze_price_trend(price_prediction, price_data),
            'recommendation': self._interpret_recommendation(recommendation),
            'final_decision': final_decision,
            'confidence_scores': {
                'sentiment': float(sentiment_confidence),
                'recommendation': float(recommendation_confidence)
            },
            'model_outputs': {
                'sentiment': sentiment_result.numpy().tolist(),
                'price_prediction': price_prediction.numpy().tolist(),
                'recommendation': recommendation.numpy().tolist()
            }
        }
        
        # Logging
        self.analysis_log.append(analysis_result)
        
        return analysis_result
    
    def _compute_weighted_decision(self,
                                  sentiment: tf.Tensor,
                                  price: tf.Tensor,
                                  recommendation: tf.Tensor) -> str:
        """Calcul de décision pondérée"""
        # Extraction des scores
        sentiment_score = tf.reduce_mean(sentiment[:, 0] - sentiment[:, 2]).numpy()
        
        # Score de prix (direction de la prédiction)
        price_direction = tf.sign(price[0, -1] - price[0, 0]).numpy()
        
        # Score de recommandation
        recommendation_score = tf.reduce_mean(recommendation[:, 0] - recommendation[:, 2]).numpy()
        
        # Score combiné
        combined_score = (
            sentiment_score * self.decision_weights['sentiment'] +
            price_direction * self.decision_weights['price'] +
            recommendation_score * self.decision_weights['fundamental']
        )
        
        # Décision basée sur le score
        if combined_score > 0.6:
            return "STRONG_BUY"
        elif combined_score > 0.3:
            return "BUY"
        elif combined_score > -0.3:
            return "HOLD"
        elif combined_score > -0.6:
            return "SELL"
        else:
            return "STRONG_SELL"
    
    def _interpret_sentiment(self, sentiment_output: tf.Tensor) -> str:
        """Interprétation du sentiment"""
        sentiment_idx = tf.argmax(sentiment_output, axis=-1).numpy()[0]
        sentiments = ["BULLISH", "NEUTRAL", "BEARISH"]
        return sentiments[sentiment_idx]
    
    def _analyze_price_trend(self,
                            prediction: tf.Tensor,
                            historical: np.ndarray) -> str:
        """Analyse de tendance des prix"""
        predicted_change = prediction[0, -1] - historical[0, -1, 0]
        if predicted_change > 0.02:  # +2%
            return "STRONG_UPTREND"
        elif predicted_change > 0:
            return "UPTREND"
        elif predicted_change > -0.02:
            return "SIDEWAYS"
        else:
            return "DOWNTREND"
    
    def _interpret_recommendation(self, recommendation: tf.Tensor) -> str:
        """Interprétation de la recommandation"""
        rec_idx = tf.argmax(recommendation, axis=-1).numpy()[0]
        recommendations = ["BUY", "HOLD", "SELL"]
        return recommendations[rec_idx]
    
    def save_analysis_report(self, filepath: str):
        """Sauvegarde des analyses"""
        report = {
            'config': self.config,
            'analyses': self.analysis_log,
            'summary': self._generate_summary()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Rapport sauvegardé: {filepath}")
    
    def _generate_summary(self) -> Dict:
        """Génère un résumé des analyses"""
        if not self.analysis_log:
            return {}
        
        decisions = [log['final_decision'] for log in self.analysis_log]
        
        return {
            'total_analyses': len(self.analysis_log),
            'decision_distribution': {
                decision: decisions.count(decision) 
                for decision in set(decisions)
            },
            'average_confidence': np.mean([
                log['confidence_scores']['sentiment'] 
                for log in self.analysis_log
            ])
        }


class ModelTrainingSystem:
    """
    Système d'entraînement coordonné pour modèles spécialisés
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
    def create_training_data(self) -> Dict:
        """Génération de données d'entraînement synthétiques"""
        batch_size = self.config['batch_size']
        
        # Données textuelles (indices de vocabulaire)
        text_data = np.random.randint(
            0, self.config['vocab_size'],
            (batch_size, self.config['seq_len'])
        )
        
        # Données de prix (séries temporelles multi-variables)
        time_steps = 30
        price_data = np.cumsum(
            np.random.randn(batch_size, time_steps, 5) * 0.01,
            axis=1
        )
        
        # Données techniques et fondamentales
        technical_data = np.random.randn(batch_size, 20)
        fundamental_data = np.random.randn(batch_size, 15)
        
        # Labels
        sentiment_labels = np.random.randint(0, 3, (batch_size,))
        price_labels = np.random.randn(batch_size, 5)
        recommendation_labels = np.random.randint(0, 3, (batch_size,))
        
        return {
            'text': text_data,
            'price': price_data,
            'technical': technical_data,
            'fundamental': fundamental_data,
            'labels': {
                'sentiment': sentiment_labels,
                'price': price_prediction_labels,
                'recommendation': recommendation_labels
            }
        }
    
    def train_models(self) -> Tuple[Model, Model, Model]:
        """Entraînement parallèle des modèles spécialisés"""
        print("Initialisation de l'entraînement...")
        
        # Création des données
        data = self.create_training_data()
        
        # Initialisation des modèles
        sentiment_model = SentimentAnalyzer(self.config)
        price_model = PricePredictor()
        recommender_model = InvestmentRecommender(self.config)
        
        # Compilation des modèles
        sentiment_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config['learning_rate']
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        price_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config['learning_rate']
            ),
            loss='mse',
            metrics=['mae']
        )
        
        recommender_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config['learning_rate']
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        # Entraînement
        print("\nEntraînement du modèle de sentiment...")
        sentiment_model.fit(
            data['text'],
            data['labels']['sentiment'],
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_split=0.2,
            verbose=1
        )
        
        print("\nEntraînement du modèle de prédiction de prix...")
        price_model.fit(
            data['price'],
            data['labels']['price'],
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_split=0.2,
            verbose=1
        )
        
        print("\nEntraînement du modèle de recommandation...")
        recommender_model.fit(
            x=[data['technical'], data['fundamental'],
               tf.one_hot(data['labels']['sentiment'], 3)],
            y=data['labels']['recommendation'],
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_split=0.2,
            verbose=1
        )
        
        return sentiment_model, price_model, recommender_model


def main():
    """
    Démonstration du système multi-modèles
    """
    print("=" * 60)
    print("SYSTÈME D'ANALYSE BOURSIÈRE - MULTI-MODÈLES")
    print("=" * 60)
    
    # 1. Initialisation du système d'entraînement
    trainer = ModelTrainingSystem(CONFIG)
    
    # 2. Entraînement des modèles
    print("\nPhase 1: Entraînement des modèles spécialisés...")
    sentiment_model, price_model, recommender_model = trainer.train_models()
    
    # 3. Initialisation de l'orchestrateur
    print("\nPhase 2: Initialisation de l'orchestrateur...")
    orchestrator = MarketAnalysisOrchestrator(CONFIG)
    
    # 4. Création de données de test
    test_data = trainer.create_training_data()
    
    # 5. Analyse de démonstration
    print("\nPhase 3: Analyse de démonstration...")
    
    analysis_result = orchestrator.analyze_market(
        text_data=test_data['text'][:2],  # Premier échantillon
        price_data=test_data['price'][:2],
        technical_data=test_data['technical'][:2],
        fundamental_data=test_data['fundamental'][:2]
    )
    
    # 6. Affichage des résultats
    print("\n" + "=" * 60)
    print("RÉSULTATS DE L'ANALYSE")
    print("=" * 60)
    print(f"Timestamp: {analysis_result['timestamp']}")
    print(f"Sentiment du marché: {analysis_result['sentiment']}")
    print(f"Tendance des prix: {analysis_result['price_trend']}")
    print(f"Recommandation: {analysis_result['recommendation']}")
    print(f"Décision finale: {analysis_result['final_decision']}")
    print(f"Confiance sentiment: {analysis_result['confidence_scores']['sentiment']:.2%}")
    print(f"Confiance recommandation: {analysis_result['confidence_scores']['recommendation']:.2%}")
    print("=" * 60)
    
    # 7. Sauvegarde du rapport
    orchestrator.save_analysis_report('market_analysis_report.json')
    
    print("\n✓ Système prêt pour l'analyse en temps réel")

if __name__ == "__main__":
    main()
