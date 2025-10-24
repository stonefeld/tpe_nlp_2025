#!/usr/bin/env python3
"""
Script de configuración e instalación del proyecto de detección de clickbait
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """
    Instalar dependencias del proyecto
    """
    print("Instalando dependencias...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencias instaladas exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error instalando dependencias: {e}")
        return False

def setup_nltk():
    """
    Configurar datos de NLTK
    """
    print("Configurando NLTK...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✓ NLTK configurado")
        return True
    except Exception as e:
        print(f"✗ Error configurando NLTK: {e}")
        return False

def create_directories():
    """
    Crear directorios necesarios
    """
    print("Creando directorios...")
    directories = ['data', 'results', 'plots']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✓ Directorios creados")
    return True

def verify_installation():
    """
    Verificar que la instalación fue exitosa
    """
    print("Verificando instalación...")
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'nltk', 
        'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"✗ Paquetes faltantes: {missing_packages}")
        return False
    else:
        print("✓ Instalación verificada")
        return True

def run_quick_test():
    """
    Ejecutar una prueba rápida del proyecto
    """
    print("Ejecutando prueba rápida...")
    try:
        # Importar módulos principales
        from clickbait_detection import ClickbaitDetector
        from download_dataset import create_sample_dataset
        
        # Crear dataset de muestra
        dataset_path = create_sample_dataset()
        
        # Inicializar detector
        detector = ClickbaitDetector()
        
        # Cargar y preprocesar datos
        import pandas as pd
        df = pd.read_csv(dataset_path)
        df['processed_text'] = df['headline'].apply(detector.preprocess_text)
        
        print("✓ Prueba rápida exitosa")
        return True
    except Exception as e:
        print(f"✗ Error en prueba rápida: {e}")
        return False

def main():
    """
    Función principal de configuración
    """
    print("CONFIGURACIÓN DEL PROYECTO DE DETECCIÓN DE CLICKBAIT")
    print("="*60)
    
    steps = [
        ("Instalando dependencias", install_requirements),
        ("Configurando NLTK", setup_nltk),
        ("Creando directorios", create_directories),
        ("Verificando instalación", verify_installation),
        ("Ejecutando prueba rápida", run_quick_test)
    ]
    
    for step_name, step_function in steps:
        print(f"\n{step_name}...")
        if not step_function():
            print(f"✗ Error en: {step_name}")
            return False
    
    print("\n" + "="*60)
    print("🎉 CONFIGURACIÓN COMPLETADA EXITOSAMENTE!")
    print("="*60)
    print("\nPara ejecutar el proyecto:")
    print("  python main.py")
    print("\nPara análisis interactivo:")
    print("  jupyter notebook clickbait_analysis.ipynb")
    print("\nPara experimentos de embeddings:")
    print("  python embeddings_experiment.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
