"""
Script para descargar el dataset de clickbait de Kaggle
"""

import os
import kaggle
import pandas as pd
from pathlib import Path

def download_clickbait_dataset():
    """
    Descargar el dataset de clickbait de Kaggle
    """
    print("Descargando dataset de clickbait de Kaggle...")
    
    # Configurar API de Kaggle (el usuario debe tener su kaggle.json configurado)
    try:
        # Descargar el dataset
        kaggle.api.dataset_download_files(
            'saurabhshahane/clickbait-dataset', 
            path='./data', 
            unzip=True
        )
        print("Dataset descargado exitosamente!")
        
        # Verificar que el archivo existe
        data_files = list(Path('./data').glob('*.csv'))
        if data_files:
            print(f"Archivos encontrados: {data_files}")
            
            # Cargar y mostrar información básica del dataset
            df = pd.read_csv(data_files[0])
            print(f"\nInformación del dataset:")
            print(f"Forma: {df.shape}")
            print(f"Columnas: {df.columns.tolist()}")
            print(f"Primeras 5 filas:")
            print(df.head())
            
            return str(data_files[0])
        else:
            print("Error: No se encontraron archivos CSV en el directorio data/")
            return None
            
    except Exception as e:
        print(f"Error al descargar el dataset: {e}")
        print("\nInstrucciones para configurar Kaggle API:")
        print("1. Ve a https://www.kaggle.com/account")
        print("2. Crea un nuevo token API")
        print("3. Descarga kaggle.json")
        print("4. Colócalo en ~/.kaggle/kaggle.json")
        print("5. Ejecuta: chmod 600 ~/.kaggle/kaggle.json")
        return None

def create_sample_dataset():
    """
    Crear un dataset de muestra para pruebas si no se puede descargar el original
    """
    print("Creando dataset de muestra...")
    
    # Dataset de muestra basado en patrones típicos de clickbait
    sample_headlines = [
        # Clickbait (1)
        "You Won't Believe What Happens Next!",
        "This One Trick Will Change Your Life Forever",
        "10 Things That Will Shock You About Celebrities",
        "What They Don't Want You to Know",
        "The Secret That Doctors Don't Want You to Know",
        "This Simple Trick Will Make You Rich",
        "You'll Never Guess What Happened Next",
        "The One Thing That Will Change Everything",
        "What Happens Next Will Amaze You",
        "This Will Blow Your Mind",
        "The Shocking Truth About...",
        "What They're Not Telling You",
        "This Changes Everything You Know",
        "You Won't Believe Your Eyes",
        "The Hidden Secret They Don't Want You to Know",
        "This One Weird Trick...",
        "What Happens When You...",
        "The Surprising Truth About...",
        "This Will Change Your Life",
        "You'll Be Shocked When You See This",
        
        # No Clickbait (0)
        "Scientists Discover New Species in Amazon",
        "Breaking: Major Earthquake Hits California",
        "New Study Shows Benefits of Exercise",
        "President Announces New Economic Policy",
        "Local School Wins National Competition",
        "New Technology Improves Solar Panel Efficiency",
        "City Council Approves New Housing Development",
        "Research Shows Link Between Diet and Health",
        "Company Reports Record Quarterly Profits",
        "University Opens New Research Center",
        "Government Announces New Environmental Initiative",
        "Study Finds Benefits of Regular Exercise",
        "New Law Aims to Reduce Traffic Accidents",
        "Scientists Develop New Cancer Treatment",
        "Local Business Expands to New Location",
        "Research Team Discovers Ancient Artifacts",
        "New Program Helps Students with Math",
        "City Plans Major Infrastructure Improvements",
        "Company Launches New Product Line",
        "Study Reveals Impact of Climate Change"
    ]
    
    # Crear etiquetas (1 para clickbait, 0 para no clickbait)
    labels = [1] * 20 + [0] * 20
    
    # Crear DataFrame
    df = pd.DataFrame({
        'headline': sample_headlines,
        'clickbait': labels
    })
    
    # Guardar dataset
    os.makedirs('./data', exist_ok=True)
    df.to_csv('./data/clickbait_sample.csv', index=False)
    
    print(f"Dataset de muestra creado: ./data/clickbait_sample.csv")
    print(f"Forma: {df.shape}")
    print(f"Distribución de clases:")
    print(df['clickbait'].value_counts())
    
    return './data/clickbait_sample.csv'

if __name__ == "__main__":
    # Intentar descargar el dataset real
    dataset_path = download_clickbait_dataset()
    
    # Si no se puede descargar, crear uno de muestra
    if dataset_path is None:
        print("\nNo se pudo descargar el dataset de Kaggle.")
        print("Creando dataset de muestra para continuar con el proyecto...")
        dataset_path = create_sample_dataset()
    
    print(f"\nDataset listo en: {dataset_path}")
