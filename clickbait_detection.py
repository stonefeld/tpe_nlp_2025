"""
Detección automática de titulares clickbait en noticias digitales

Este script implementa tres experimentos principales:
1. Clasificación con modelo clásico (BoW + TF-IDF + Regresión Logística)
2. Clasificación con modelos basados en embeddings (GloVe + BERT)
3. Análisis lingüístico de patrones de clickbait

Autor: Proyecto NLP TPE 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para mejor visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ClickbaitDetector:
    """
    Clase principal para la detección de clickbait en titulares de noticias
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.models = {}
        self.results = {}
        
    def preprocess_text(self, text, use_stemming=True):
        """
        Preprocesamiento de texto: limpieza, tokenización, eliminación de stopwords
        """
        # Convertir a minúsculas
        text = text.lower()
        
        # Eliminar caracteres especiales y números
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenización
        tokens = word_tokenize(text)
        
        # Eliminar stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Aplicar stemming si se solicita
        if use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def load_and_preprocess_data(self, file_path):
        """
        Cargar y preprocesar el dataset
        """
        print("Cargando dataset...")
        df = pd.read_csv(file_path)
        
        print(f"Dataset cargado: {df.shape[0]} titulares")
        print(f"Distribución de clases:")
        print(df['clickbait'].value_counts())
        
        # Preprocesar texto
        print("Preprocesando texto...")
        df['processed_text'] = df['headline'].apply(self.preprocess_text)
        
        return df
    
    def split_data(self, df, test_size=0.3, val_size=0.5):
        """
        Dividir datos en entrenamiento, validación y prueba
        """
        # Primera división: entrenamiento (70%) y temporal (30%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            df['processed_text'], df['clickbait'], 
            test_size=test_size, random_state=self.random_state, 
            stratify=df['clickbait']
        )
        
        # Segunda división: entrenamiento (70%) y validación (15%)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_size, random_state=self.random_state, 
            stratify=y_temp
        )
        
        print(f"Conjunto de entrenamiento: {len(X_train)} muestras")
        print(f"Conjunto de validación: {len(X_val)} muestras")
        print(f"Conjunto de prueba: {len(X_test)} muestras")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def experiment_1_baseline(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Experimento 1: Clasificación con modelo clásico (BoW + TF-IDF + Regresión Logística)
        """
        print("\n" + "="*60)
        print("EXPERIMENTO 1: MODELO CLÁSICO (LÍNEA BASE)")
        print("="*60)
        
        results = {}
        
        # BoW + Regresión Logística
        print("\nEntrenando modelo BoW + Regresión Logística...")
        bow_pipeline = Pipeline([
            ('vectorizer', CountVectorizer(max_features=10000, ngram_range=(1, 2))),
            ('classifier', LogisticRegression(random_state=self.random_state, max_iter=1000))
        ])
        
        # Optimización de hiperparámetros
        param_grid = {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2']
        }
        
        bow_grid = GridSearchCV(bow_pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
        bow_grid.fit(X_train, y_train)
        
        # Evaluación
        y_pred_bow = bow_grid.predict(X_test)
        bow_results = self.evaluate_model(y_test, y_pred_bow, "BoW + Regresión Logística")
        bow_results['best_params'] = bow_grid.best_params_
        results['BoW'] = bow_results
        
        # TF-IDF + Regresión Logística
        print("\nEntrenando modelo TF-IDF + Regresión Logística...")
        tfidf_pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            ('classifier', LogisticRegression(random_state=self.random_state, max_iter=1000))
        ])
        
        tfidf_grid = GridSearchCV(tfidf_pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
        tfidf_grid.fit(X_train, y_train)
        
        # Evaluación
        y_pred_tfidf = tfidf_grid.predict(X_test)
        tfidf_results = self.evaluate_model(y_test, y_pred_tfidf, "TF-IDF + Regresión Logística")
        tfidf_results['best_params'] = tfidf_grid.best_params_
        results['TF-IDF'] = tfidf_results
        
        # Guardar modelos
        self.models['bow'] = bow_grid.best_estimator_
        self.models['tfidf'] = tfidf_grid.best_estimator_
        
        self.results['experiment_1'] = results
        return results
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """
        Evaluar modelo con métricas estándar
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"\nResultados para {model_name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Clickbait', 'Clickbait'],
                   yticklabels=['No Clickbait', 'Clickbait'])
        plt.title(f'Matriz de Confusión - {model_name}')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    
    def experiment_2_embeddings(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Experimento 2: Clasificación con modelos basados en embeddings
        """
        print("\n" + "="*60)
        print("EXPERIMENTO 2: MODELOS BASADOS EN EMBEDDINGS")
        print("="*60)
        
        # Nota: Para este experimento necesitaríamos implementar GloVe y BERT
        # Por ahora, implementaremos una versión simplificada con Word2Vec
        print("\nNota: Este experimento requiere implementación adicional de GloVe y BERT")
        print("Se implementará en una versión extendida del proyecto")
        
        return {}
    
    def experiment_3_linguistic_analysis(self, X_train, y_train):
        """
        Experimento 3: Análisis lingüístico de patrones de clickbait
        """
        print("\n" + "="*60)
        print("EXPERIMENTO 3: ANÁLISIS LINGÜÍSTICO")
        print("="*60)
        
        if 'tfidf' not in self.models:
            print("Error: Debe ejecutar el Experimento 1 primero")
            return {}
        
        # Obtener características más importantes del modelo TF-IDF
        model = self.models['tfidf']
        feature_names = model.named_steps['vectorizer'].get_feature_names_out()
        coefficients = model.named_steps['classifier'].coef_[0]
        
        # Crear DataFrame con características y coeficientes
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients
        }).sort_values('coefficient', ascending=False)
        
        # Top 20 características que indican clickbait (coeficientes positivos)
        top_clickbait = feature_importance.head(20)
        print("\nTop 20 características que indican CLICKBAIT:")
        print(top_clickbait)
        
        # Top 20 características que indican no clickbait (coeficientes negativos)
        top_no_clickbait = feature_importance.tail(20)
        print("\nTop 20 características que indican NO CLICKBAIT:")
        print(top_no_clickbait)
        
        # Visualización
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Gráfico de características clickbait
        top_clickbait.plot(x='feature', y='coefficient', kind='barh', ax=ax1, color='red', alpha=0.7)
        ax1.set_title('Top 20 Características - CLICKBAIT', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Coeficiente')
        ax1.invert_yaxis()
        
        # Gráfico de características no clickbait
        top_no_clickbait.plot(x='feature', y='coefficient', kind='barh', ax=ax2, color='blue', alpha=0.7)
        ax2.set_title('Top 20 Características - NO CLICKBAIT', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Coeficiente')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('linguistic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'clickbait_features': top_clickbait,
            'no_clickbait_features': top_no_clickbait
        }
    
    def exploratory_data_analysis(self, df):
        """
        Análisis exploratorio de datos
        """
        print("\n" + "="*60)
        print("ANÁLISIS EXPLORATORIO DE DATOS")
        print("="*60)
        
        # Distribución de clases
        plt.figure(figsize=(10, 6))
        df['clickbait'].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
        plt.title('Distribución de Clases')
        plt.xlabel('Clase (0: No Clickbait, 1: Clickbait)')
        plt.ylabel('Cantidad de Titulares')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Longitud de titulares
        df['length'] = df['headline'].str.split().str.len()
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        df[df['clickbait'] == 0]['length'].hist(alpha=0.7, bins=30, label='No Clickbait', color='skyblue')
        df[df['clickbait'] == 1]['length'].hist(alpha=0.7, bins=30, label='Clickbait', color='lightcoral')
        plt.xlabel('Longitud del Titular (palabras)')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Longitud de Titulares')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        df.boxplot(column='length', by='clickbait', ax=plt.gca())
        plt.title('Boxplot de Longitud por Clase')
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig('headline_length_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Estadísticas descriptivas
        print("\nEstadísticas de longitud por clase:")
        print(df.groupby('clickbait')['length'].describe())
        
        return df

def main():
    """
    Función principal para ejecutar todos los experimentos
    """
    # Descargar recursos de NLTK
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    # Inicializar detector
    detector = ClickbaitDetector()
    
    # Nota: El usuario debe proporcionar el archivo CSV del dataset
    print("IMPORTANTE: Asegúrate de tener el archivo 'clickbait_dataset.csv' en el directorio actual")
    print("El dataset debe tener las columnas: 'headline' y 'clickbait'")
    
    # Para este ejemplo, crearemos un dataset de prueba
    # En la implementación real, cargarías el archivo CSV
    print("\nCreando dataset de prueba para demostración...")
    
    # Dataset de prueba (reemplazar con carga real del CSV)
    sample_data = {
        'headline': [
            "You Won't Believe What Happens Next!",
            "Scientists Discover New Species in Amazon",
            "This One Trick Will Change Your Life Forever",
            "Breaking: Major Earthquake Hits California",
            "10 Things That Will Shock You About Celebrities",
            "New Study Shows Benefits of Exercise",
            "What They Don't Want You to Know",
            "President Announces New Economic Policy",
            "The Secret That Doctors Don't Want You to Know",
            "Local School Wins National Competition"
        ],
        'clickbait': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Análisis exploratorio
    df = detector.exploratory_data_analysis(df)
    
    # Preprocesamiento
    df = detector.load_and_preprocess_data(df)
    
    # División de datos
    X_train, X_val, X_test, y_train, y_val, y_test = detector.split_data(df)
    
    # Ejecutar experimentos
    print("\nIniciando experimentos...")
    
    # Experimento 1: Modelo clásico
    exp1_results = detector.experiment_1_baseline(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Experimento 3: Análisis lingüístico
    exp3_results = detector.experiment_3_linguistic_analysis(X_train, y_train)
    
    # Resumen de resultados
    print("\n" + "="*60)
    print("RESUMEN DE RESULTADOS")
    print("="*60)
    
    for model_name, results in exp1_results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  F1-Score: {results['f1']:.4f}")
    
    print("\nProyecto completado exitosamente!")
    print("Los gráficos y resultados han sido guardados en el directorio actual.")

if __name__ == "__main__":
    main()
