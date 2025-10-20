#!/usr/bin/env python3
"""
Script principal para ejecutar el proyecto de detección de clickbait
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path

def check_dependencies():
    """
    Verificar que las dependencias estén instaladas
    """
    print("Verificando dependencias...")
    try:
        import pandas
        import numpy
        import sklearn
        import nltk
        import matplotlib
        import seaborn
        print("✓ Todas las dependencias están instaladas")
        return True
    except ImportError as e:
        print(f"✗ Dependencia faltante: {e}")
        print("Ejecuta: pip install -r requirements.txt")
        return False

def setup_nltk_data():
    """
    Configurar datos de NLTK
    """
    print("Configurando datos de NLTK...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✓ Datos de NLTK configurados")
        return True
    except Exception as e:
        print(f"✗ Error configurando NLTK: {e}")
        return False

def download_dataset():
    """
    Descargar o crear dataset
    """
    print("Preparando dataset...")
    
    # Verificar si ya existe un dataset
    data_dir = Path('./data')
    if data_dir.exists() and list(data_dir.glob('*.csv')):
        print("✓ Dataset ya existe")
        return True
    
    # Intentar descargar dataset real
    try:
        from download_dataset import download_clickbait_dataset, create_sample_dataset
        dataset_path = download_clickbait_dataset()
        if dataset_path is None:
            dataset_path = create_sample_dataset()
        print(f"✓ Dataset preparado: {dataset_path}")
        return True
    except Exception as e:
        print(f"✗ Error preparando dataset: {e}")
        return False

def run_analysis():
    """
    Ejecutar análisis completo
    """
    print("\n" + "="*60)
    print("INICIANDO ANÁLISIS DE CLICKBAIT")
    print("="*60)
    
    try:
        # Importar y ejecutar el detector
        from clickbait_detection import ClickbaitDetector
        
        # Inicializar detector
        detector = ClickbaitDetector()
        
        # Cargar dataset
        data_files = list(Path('./data').glob('*.csv'))
        if not data_files:
            print("Error: No se encontró dataset")
            return False
        
        dataset_path = data_files[0]
        print(f"Cargando dataset: {dataset_path}")
        
        # Cargar datos
        df = pd.read_csv(dataset_path)
        print(f"Dataset cargado: {df.shape[0]} titulares")
        
        # Análisis exploratorio
        print("\nRealizando análisis exploratorio...")
        df = detector.exploratory_data_analysis(df)
        
        # Preprocesamiento
        print("\nPreprocesando texto...")
        df = detector.load_and_preprocess_data(df)
        
        # División de datos
        print("\nDividiendo datos...")
        X_train, X_val, X_test, y_train, y_val, y_test = detector.split_data(df)
        
        # Experimento 1: Modelo clásico
        print("\nEjecutando Experimento 1: Modelo clásico...")
        exp1_results = detector.experiment_1_baseline(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # Experimento 3: Análisis lingüístico
        print("\nEjecutando Experimento 3: Análisis lingüístico...")
        exp3_results = detector.experiment_3_linguistic_analysis(X_train, y_train)
        
        # Resumen final
        print("\n" + "="*60)
        print("RESUMEN FINAL")
        print("="*60)
        
        for model_name, results in exp1_results.items():
            print(f"\n{model_name}:")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  Recall: {results['recall']:.4f}")
            print(f"  F1-Score: {results['f1']:.4f}")
        
        print("\n✓ Análisis completado exitosamente!")
        print("Los gráficos y resultados han sido guardados en el directorio actual.")
        
        return True
        
    except Exception as e:
        print(f"✗ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Función principal
    """
    print("PROYECTO: Detección Automática de Titulares Clickbait")
    print("="*60)
    
    # Verificar dependencias
    if not check_dependencies():
        return False
    
    # Configurar NLTK
    if not setup_nltk_data():
        return False
    
    # Preparar dataset
    if not download_dataset():
        return False
    
    # Ejecutar análisis
    if not run_analysis():
        return False
    
    print("\n🎉 Proyecto completado exitosamente!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
