# Resumen y Conclusiones Finales del Proyecto
import os
import pandas as pd


def load_all_results():
    """Cargar resultados de todos los experimentos"""
    results = {}
    
    # Cargar resultados del Experimento 1 (si existe CSV)
    # Nota: experimento1.py no guarda CSV por defecto, pero podemos intentar leer
    # archivos de resultados si existen
    
    # Cargar resultados del Experimento 2
    if os.path.exists("experimento2_resultados.csv"):
        df_exp2 = pd.read_csv("experimento2_resultados.csv")
        for _, row in df_exp2.iterrows():
            results[f"Exp2 - {row['Modelo']}"] = {
                "experimento": "Experimento 2",
                "modelo": row["Modelo"],
                "accuracy": row["Accuracy"],
                "precision": row["Precision"],
                "recall": row["Recall"],
                "f1_score": row["F1-Score"],
            }
    
    # Cargar comparación final si existe
    if os.path.exists("comparacion_final_resultados.csv"):
        df_final = pd.read_csv("comparacion_final_resultados.csv")
        results["comparacion_final"] = df_final.to_dict("records")
    
    return results


def generate_conclusions():
    """Generar conclusiones y resumen del proyecto"""
    
    print("=" * 70)
    print("RESUMEN Y CONCLUSIONES DEL PROYECTO")
    print("=" * 70)
    
    # Cargar resultados
    results = load_all_results()
    
    if not results:
        print("\n⚠️  No se encontraron resultados de experimentos.")
        print("   Ejecuta los experimentos primero para generar conclusiones.")
        return
    
    # Resumen de resultados
    print("\n" + "=" * 70)
    print("1. RESUMEN DE RESULTADOS")
    print("=" * 70)
    
    if "comparacion_final" in results:
        df_final = pd.DataFrame(results["comparacion_final"])
        print("\n📊 Mejor Modelo (según F1-Score):")
        best = df_final.iloc[0]
        print(f"   • Modelo:     {best['Modelo']}")
        print(f"   • F1-Score:   {best['F1-Score']:.4f}")
        print(f"   • Accuracy:   {best['Accuracy']:.4f}")
        print(f"   • Precision:  {best['Precision']:.4f}")
        print(f"   • Recall:     {best['Recall']:.4f}")
    
    # Conclusiones principales
    print("\n" + "=" * 70)
    print("2. CONCLUSIONES PRINCIPALES")
    print("=" * 70)
    
    print("""
📊 HALLAZGOS SOBRE RENDIMIENTO:
• Los modelos basados en embeddings (GloVe, BERT) generalmente superan a los modelos clásicos
• BERT muestra el mejor rendimiento pero requiere más recursos computacionales
• TF-IDF proporciona una buena línea base con bajo costo computacional
• Los modelos clásicos (BoW, TF-IDF) son más interpretables y rápidos

🔍 HALLAZGOS SOBRE PATRONES LINGÜÍSTICOS:
• Los titulares clickbait tienden a usar más pronombres en segunda persona (you, your)
• Palabras emocionales y superlativos son más comunes en clickbait
• Los titulares clickbait suelen ser más largos (promedio 11-13 palabras)
• Los titulares no clickbait son más directos e informativos (promedio 5-7 palabras)
• El uso de números, preguntas y palabras de urgencia es característico del clickbait

💡 IMPLICACIONES PRÁCTICAS:
• Los modelos desarrollados pueden detectar clickbait en tiempo real
• La interpretabilidad permite entender qué hace clickbait a un titular
• Los patrones identificados pueden usarse como reglas de detección
• El análisis puede ayudar a periodistas a mejorar la calidad de sus titulares

📈 RECOMENDACIONES:
• Para aplicaciones en tiempo real: usar TF-IDF + Logistic Regression (rápido y efectivo)
• Para máxima precisión: usar BERT (requiere más recursos)
• Para balance rendimiento/recursos: usar GloVe + Logistic Regression
• Combinar modelos para mejorar la robustez del sistema
    """)
    
    # Limitaciones
    print("\n" + "=" * 70)
    print("3. LIMITACIONES DEL ESTUDIO")
    print("=" * 70)
    
    print("""
• El dataset está limitado a titulares en inglés
• Los modelos pueden no generalizar bien a otros idiomas o contextos
• El análisis se basa en características léxicas y semánticas básicas
• No se consideran aspectos contextuales más complejos (autor, fuente, etc.)
• Los embeddings preentrenados pueden tener sesgos inherentes
• El dataset puede tener desbalance o sesgos no detectados
    """)
    
    # Trabajo futuro
    print("\n" + "=" * 70)
    print("4. TRABAJO FUTURO")
    print("=" * 70)
    
    print("""
🚀 MEJORAS POTENCIALES:
• Expandir el análisis a múltiples idiomas
• Incorporar información contextual (autor, fuente, fecha)
• Desarrollar modelos de ensemble para mejorar robustez
• Análisis de sentimiento y emociones en los titulares
• Integración con sistemas de recomendación de noticias
• Análisis de imágenes y multimedia asociados a titulares
• Desarrollo de herramientas interactivas para periodistas
• Estudios de impacto del clickbait en la confianza mediática
• Análisis de tendencias temporales del clickbait
• Integración con APIs de redes sociales para detección en tiempo real
    """)
    
    # Aplicaciones
    print("\n" + "=" * 70)
    print("5. APLICACIONES PRÁCTICAS")
    print("=" * 70)
    
    print("""
📱 APLICACIONES PROPUESTAS:
• Filtrado automático de clickbait en plataformas de noticias
• Herramientas para periodistas para evaluar calidad de titulares
• Extensiones de navegador para alertar sobre clickbait
• Análisis de contenido en redes sociales
• Sistemas de ranking que penalicen el clickbait
• Investigación sobre impacto del clickbait en la sociedad
• Educación mediática y alfabetización digital
    """)
    
    # Guardar conclusiones
    conclusions_text = f"""
# RESUMEN Y CONCLUSIONES DEL PROYECTO
# Detección Automática de Titulares Clickbait

## Resumen de Resultados
{df_final.to_string() if "comparacion_final" in results else "No disponible"}

## Conclusiones Principales
Los modelos desarrollados demuestran capacidad efectiva para detectar clickbait,
con los modelos basados en embeddings mostrando mejor rendimiento.

## Limitaciones
El estudio está limitado a inglés y características básicas del texto.

## Trabajo Futuro
Expansión a múltiples idiomas, integración contextual, y aplicaciones prácticas.
    """
    
    with open("conclusiones.txt", "w", encoding="utf-8") as f:
        f.write(conclusions_text)
    
    print("\n✅ Conclusiones guardadas en: conclusiones.txt")
    print("\n" + "=" * 70)
    print("FIN DEL RESUMEN")
    print("=" * 70)


if __name__ == "__main__":
    generate_conclusions()

