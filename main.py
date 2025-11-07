#!/usr/bin/env python3
"""
Script Principal - Ejecuta todos los experimentos y análisis del proyecto
Detección Automática de Titulares Clickbait en Noticias Digitales
"""
import os
import sys
import subprocess
from pathlib import Path


def check_dataset():
    """Verificar que el dataset existe"""
    dataset_path = Path("assets/clickbait_data.csv")
    if not dataset_path.exists():
        print("❌ Error: No se encontró el dataset en assets/clickbait_data.csv")
        print("   Asegúrate de que el archivo existe antes de ejecutar los experimentos.")
        return False
    print(f"✅ Dataset encontrado: {dataset_path}")
    return True


def check_dependencies():
    """Verificar dependencias básicas"""
    print("\n🔍 Verificando dependencias...")
    try:
        import pandas
        import numpy
        import sklearn
        import matplotlib
        import seaborn
        import nltk
        print("✅ Dependencias base instaladas")
        return True
    except ImportError as e:
        print(f"❌ Dependencia faltante: {e}")
        print("   Instala las dependencias: pip install -r requirements.txt")
        return False


def run_script(script_name, description):
    """Ejecutar un script de Python"""
    script_path = Path(script_name)
    if not script_path.exists():
        print(f"⚠️  Advertencia: {script_name} no encontrado. Saltando...")
        return False
    
    print(f"\n{'='*70}")
    print(f"EJECUTANDO: {description}")
    print(f"{'='*70}")
    
    try:
        # Ejecutar el script
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✅ {description} completado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error ejecutando {script_name}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  {description} interrumpido por el usuario")
        return False
    except Exception as e:
        print(f"\n❌ Error inesperado en {script_name}: {e}")
        return False


def main():
    """Función principal"""
    print("=" * 70)
    print("PROYECTO: Detección Automática de Titulares Clickbait")
    print("Análisis Completo con NLP")
    print("=" * 70)
    
    # Verificar dependencias
    if not check_dependencies():
        sys.exit(1)
    
    # Verificar dataset
    if not check_dataset():
        sys.exit(1)
    
    # Lista de experimentos a ejecutar
    experiments = [
        ("eda.py", "Análisis Exploratorio de Datos (EDA)"),
        ("experimento1.py", "Experimento 1: Modelos Clásicos (BoW, TF-IDF)"),
        ("experimento2.py", "Experimento 2: Modelos Basados en Embeddings (GloVe, BERT)"),
        ("experimento3.py", "Experimento 3: Análisis Lingüístico"),
        ("comparacion_final.py", "Comparación Final de Todos los Modelos"),
        ("resumen_y_conclusiones.py", "Resumen y Conclusiones del Proyecto"),
    ]
    
    results = {}
    
    # Ejecutar cada experimento
    for script, description in experiments:
        success = run_script(script, description)
        results[description] = success
        
        if not success:
            print(f"\n⚠️  Advertencia: {description} falló o fue saltado")
            response = input("¿Deseas continuar con los siguientes experimentos? (s/n): ")
            if response.lower() != 's':
                print("\n⚠️  Ejecución cancelada por el usuario")
                break
    
    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN DE EJECUCIÓN")
    print("=" * 70)
    
    successful = sum(1 for v in results.values() if v)
    total = len(results)
    
    for description, success in results.items():
        status = "✅ Completado" if success else "❌ Falló/Saltado"
        print(f"  {status}: {description}")
    
    print(f"\n📊 Progreso: {successful}/{total} experimentos completados")
    
    if successful == total:
        print("\n🎉 ¡Todos los experimentos se completaron exitosamente!")
        print("\n📁 Archivos generados:")
        print("   • Visualizaciones PNG en el directorio actual")
        print("   • Archivos CSV con resultados")
        print("   • Comparación final de modelos")
    elif successful > 0:
        print(f"\n⚠️  {total - successful} experimentos fallaron o fueron saltados")
        print("   Revisa los errores arriba para más detalles")
    else:
        print("\n❌ Ningún experimento se completó exitosamente")
        print("   Revisa los errores arriba y las dependencias")
    
    print("\n" + "=" * 70)
    print("FIN DEL ANÁLISIS")
    print("=" * 70)
    
    return successful == total


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Ejecución interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

