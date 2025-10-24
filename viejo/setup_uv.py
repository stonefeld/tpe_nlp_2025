#!/usr/bin/env python3
"""
Script de configuración e instalación del proyecto usando uv
"""

import subprocess
import sys
import os
from pathlib import Path

def check_uv_installed():
    """
    Verificar si uv está instalado
    """
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        print("✓ uv está instalado")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ uv no está instalado")
        print("Instala uv con: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False

def install_dependencies():
    """
    Instalar dependencias básicas con uv
    """
    print("Instalando dependencias básicas...")
    try:
        subprocess.check_call(["uv", "sync"])
        print("✓ Dependencias básicas instaladas")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error instalando dependencias: {e}")
        return False

def install_optional_dependencies():
    """
    Instalar dependencias opcionales
    """
    print("Instalando dependencias opcionales...")
    
    # Embeddings (Word2Vec, BERT, etc.)
    try:
        subprocess.check_call(["uv", "sync", "--extra", "embeddings"])
        print("✓ Dependencias de embeddings instaladas")
    except subprocess.CalledProcessError as e:
        print(f"⚠ No se pudieron instalar dependencias de embeddings: {e}")
    
    # Kaggle (opcional)
    try:
        subprocess.check_call(["uv", "sync", "--extra", "kaggle"])
        print("✓ Dependencias de Kaggle instaladas")
    except subprocess.CalledProcessError as e:
        print(f"⚠ No se pudieron instalar dependencias de Kaggle: {e}")

def setup_nltk():
    """
    Configurar datos de NLTK
    """
    print("Configurando NLTK...")
    try:
        # Ejecutar con el entorno de uv
        subprocess.check_call([
            "uv", "run", "python", "-c", 
            "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
        ])
        print("✓ NLTK configurado")
        return True
    except subprocess.CalledProcessError as e:
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
    try:
        subprocess.check_call([
            "uv", "run", "python", "-c",
            "import pandas, numpy, sklearn, nltk, matplotlib, seaborn; print('✓ Todas las dependencias están disponibles')"
        ])
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error verificando instalación: {e}")
        return False

def run_quick_test():
    """
    Ejecutar una prueba rápida del proyecto
    """
    print("Ejecutando prueba rápida...")
    try:
        subprocess.check_call([
            "uv", "run", "python", "-c",
            """
from clickbait_detection import ClickbaitDetector
from download_dataset import create_sample_dataset
import pandas as pd

# Crear dataset de muestra
dataset_path = create_sample_dataset()

# Inicializar detector
detector = ClickbaitDetector()

# Cargar y preprocesar datos
df = pd.read_csv(dataset_path)
df['processed_text'] = df['headline'].apply(detector.preprocess_text)

print('✓ Prueba rápida exitosa')
            """
        ])
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error en prueba rápida: {e}")
        return False

def create_uv_scripts():
    """
    Crear scripts de conveniencia para uv
    """
    print("Creando scripts de conveniencia...")
    
    # Script para ejecutar el análisis principal
    with open("run_analysis.py", "w") as f:
        f.write('''#!/usr/bin/env python3
"""
Script para ejecutar el análisis principal con uv
"""
import subprocess
import sys

def main():
    try:
        subprocess.check_call(["uv", "run", "python", "main.py"])
    except subprocess.CalledProcessError as e:
        print(f"Error ejecutando análisis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
''')
    
    # Script para ejecutar la demostración
    with open("run_demo.py", "w") as f:
        f.write('''#!/usr/bin/env python3
"""
Script para ejecutar la demostración con uv
"""
import subprocess
import sys

def main():
    try:
        subprocess.check_call(["uv", "run", "python", "demo.py"])
    except subprocess.CalledProcessError as e:
        print(f"Error ejecutando demostración: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
''')
    
    # Script para ejecutar experimentos de embeddings
    with open("run_embeddings.py", "w") as f:
        f.write('''#!/usr/bin/env python3
"""
Script para ejecutar experimentos de embeddings con uv
"""
import subprocess
import sys

def main():
    try:
        subprocess.check_call(["uv", "run", "python", "embeddings_experiment.py"])
    except subprocess.CalledProcessError as e:
        print(f"Error ejecutando experimentos de embeddings: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
''')
    
    # Hacer los scripts ejecutables
    for script in ["run_analysis.py", "run_demo.py", "run_embeddings.py"]:
        os.chmod(script, 0o755)
    
    print("✓ Scripts de conveniencia creados")

def main():
    """
    Función principal de configuración con uv
    """
    print("CONFIGURACIÓN DEL PROYECTO CON UV")
    print("="*50)
    
    steps = [
        ("Verificando uv", check_uv_installed),
        ("Creando directorios", create_directories),
        ("Instalando dependencias básicas", install_dependencies),
        ("Instalando dependencias opcionales", install_optional_dependencies),
        ("Configurando NLTK", setup_nltk),
        ("Verificando instalación", verify_installation),
        ("Creando scripts de conveniencia", create_uv_scripts),
        ("Ejecutando prueba rápida", run_quick_test)
    ]
    
    for step_name, step_function in steps:
        print(f"\n{step_name}...")
        if not step_function():
            if step_name == "Verificando uv":
                return False
            print(f"⚠ Continuando sin: {step_name}")
    
    print("\n" + "="*50)
    print("🎉 CONFIGURACIÓN CON UV COMPLETADA!")
    print("="*50)
    print("\nComandos disponibles:")
    print("  uv run python main.py              # Análisis completo")
    print("  uv run python demo.py              # Demostración interactiva")
    print("  uv run python embeddings_experiment.py  # Experimentos de embeddings")
    print("  uv run jupyter notebook clickbait_analysis.ipynb  # Análisis interactivo")
    print("\nScripts de conveniencia:")
    print("  python run_analysis.py             # Análisis completo")
    print("  python run_demo.py                 # Demostración")
    print("  python run_embeddings.py           # Experimentos de embeddings")
    print("\nPara instalar dependencias adicionales:")
    print("  uv sync --extra embeddings         # Para Word2Vec, BERT, etc.")
    print("  uv sync --extra kaggle             # Para descargar datasets de Kaggle")
    print("  uv sync --extra dev                # Para desarrollo")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
