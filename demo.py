#!/usr/bin/env python3
"""
Script de demostración del proyecto de detección de clickbait
Muestra ejemplos de clasificación y análisis de patrones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from clickbait_detection import ClickbaitDetector
from download_dataset import create_sample_dataset
import warnings
warnings.filterwarnings('ignore')

def demonstrate_classification():
    """
    Demostrar clasificación de titulares
    """
    print("DEMOSTRACIÓN: Clasificación de Titulares")
    print("="*50)
    
    # Crear dataset de muestra
    detector = ClickbaitDetector()
    dataset_path = create_sample_dataset()
    df = pd.read_csv(dataset_path)
    
    # Preprocesar
    df['processed_text'] = df['headline'].apply(detector.preprocess_text)
    
    # Dividir datos
    X_train, X_val, X_test, y_train, y_val, y_test = detector.split_data(df)
    
    # Entrenar modelo TF-IDF (más rápido para demo)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Ejemplos de clasificación
    test_examples = [
        "You Won't Believe What Happens Next!",
        "Scientists Discover New Species in Amazon",
        "This One Trick Will Change Your Life Forever",
        "Breaking: Major Earthquake Hits California",
        "10 Things That Will Shock You About Celebrities",
        "New Study Shows Benefits of Exercise",
        "What They Don't Want You to Know",
        "President Announces New Economic Policy"
    ]
    
    print("\nEjemplos de Clasificación:")
    print("-" * 50)
    
    for headline in test_examples:
        processed = detector.preprocess_text(headline)
        prediction = pipeline.predict([processed])[0]
        probability = pipeline.predict_proba([processed])[0]
        
        label = "CLICKBAIT" if prediction == 1 else "NO CLICKBAIT"
        confidence = max(probability) * 100
        
        print(f"Titular: {headline}")
        print(f"Predicción: {label} (Confianza: {confidence:.1f}%)")
        print("-" * 50)
    
    return pipeline, detector

def demonstrate_pattern_analysis(pipeline, detector):
    """
    Demostrar análisis de patrones lingüísticos
    """
    print("\nDEMOSTRACIÓN: Análisis de Patrones Lingüísticos")
    print("="*50)
    
    # Obtener características más importantes
    feature_names = pipeline.named_steps['vectorizer'].get_feature_names_out()
    coefficients = pipeline.named_steps['classifier'].coef_[0]
    
    # Crear DataFrame con características y coeficientes
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    }).sort_values('coefficient', ascending=False)
    
    # Top características clickbait
    top_clickbait = feature_importance.head(10)
    print("\nTop 10 características que indican CLICKBAIT:")
    for i, (_, row) in enumerate(top_clickbait.iterrows(), 1):
        print(f"{i:2d}. {row['feature']:20s} (coef: {row['coefficient']:+.3f})")
    
    # Top características no clickbait
    top_no_clickbait = feature_importance.tail(10)
    print("\nTop 10 características que indican NO CLICKBAIT:")
    for i, (_, row) in enumerate(top_no_clickbait.iterrows(), 1):
        print(f"{i:2d}. {row['feature']:20s} (coef: {row['coefficient']:+.3f})")
    
    # Visualización
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Gráfico clickbait
    top_clickbait.plot(x='feature', y='coefficient', kind='barh', ax=ax1, color='red', alpha=0.7)
    ax1.set_title('Top 10 Características - CLICKBAIT', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Coeficiente')
    ax1.invert_yaxis()
    
    # Gráfico no clickbait
    top_no_clickbait.plot(x='feature', y='coefficient', kind='barh', ax=ax2, color='blue', alpha=0.7)
    ax2.set_title('Top 10 Características - NO CLICKBAIT', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Coeficiente')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('demo_pattern_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_interactive_classification():
    """
    Demostrar clasificación interactiva
    """
    print("\nDEMOSTRACIÓN: Clasificación Interactiva")
    print("="*50)
    print("Ingresa titulares para clasificar (escribe 'quit' para salir):")
    
    # Entrenar modelo rápido
    detector = ClickbaitDetector()
    dataset_path = create_sample_dataset()
    df = pd.read_csv(dataset_path)
    df['processed_text'] = df['headline'].apply(detector.preprocess_text)
    
    X_train, X_val, X_test, y_train, y_val, y_test = detector.split_data(df)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    pipeline.fit(X_train, y_train)
    
    while True:
        headline = input("\nTitular: ").strip()
        if headline.lower() == 'quit':
            break
        
        if headline:
            processed = detector.preprocess_text(headline)
            prediction = pipeline.predict([processed])[0]
            probability = pipeline.predict_proba([processed])[0]
            
            label = "CLICKBAIT" if prediction == 1 else "NO CLICKBAIT"
            confidence = max(probability) * 100
            
            print(f"Predicción: {label} (Confianza: {confidence:.1f}%)")
            
            # Mostrar características más importantes
            feature_names = pipeline.named_steps['vectorizer'].get_feature_names_out()
            coefficients = pipeline.named_steps['classifier'].coef_[0]
            
            # Obtener características presentes en el texto
            words = processed.split()
            present_features = [(word, coefficients[list(feature_names).index(word)]) 
                              for word in words if word in feature_names]
            present_features.sort(key=lambda x: abs(x[1]), reverse=True)
            
            if present_features:
                print("Características más importantes en este titular:")
                for word, coef in present_features[:5]:
                    direction = "→ CLICKBAIT" if coef > 0 else "→ NO CLICKBAIT"
                    print(f"  {word:15s} {direction} (coef: {coef:+.3f})")

def main():
    """
    Función principal de demostración
    """
    print("DEMOSTRACIÓN DEL PROYECTO DE DETECCIÓN DE CLICKBAIT")
    print("="*60)
    
    try:
        # Demostrar clasificación
        pipeline, detector = demonstrate_classification()
        
        # Demostrar análisis de patrones
        demonstrate_pattern_analysis(pipeline, detector)
        
        # Demostrar clasificación interactiva
        demonstrate_interactive_classification()
        
        print("\n" + "="*60)
        print("🎉 DEMOSTRACIÓN COMPLETADA!")
        print("="*60)
        print("\nPara más análisis, ejecuta:")
        print("  python main.py              # Análisis completo")
        print("  jupyter notebook clickbait_analysis.ipynb  # Análisis interactivo")
        print("  python embeddings_experiment.py           # Experimentos avanzados")
        
    except Exception as e:
        print(f"Error en la demostración: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
