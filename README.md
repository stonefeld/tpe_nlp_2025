# Detección Automática de Titulares Clickbait en Noticias Digitales

Proyecto de NLP para detectar titulares clickbait utilizando técnicas de procesamiento de lenguaje natural y machine learning.

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema de clasificación automática para detectar titulares clickbait en noticias digitales utilizando técnicas de procesamiento de lenguaje natural (NLP). El objetivo es distinguir entre titulares engañosos diseñados para atraer clics y titulares informativos o neutrales.

## 🎯 Objetivos

1. **Clasificación binaria**: Entrenar modelos para clasificar titulares como clickbait o no clickbait
2. **Análisis comparativo**: Evaluar diferentes representaciones de texto (BoW, TF-IDF, embeddings)
3. **Interpretabilidad**: Identificar patrones lingüísticos característicos del clickbait

## 📊 Metodología

### Experimento 1: Modelo Clásico (Línea Base)
- **Representaciones**: Bag of Words (BoW) y TF-IDF
- **Algoritmo**: Regresión Logística
- **Objetivo**: Establecer línea base de rendimiento

### Experimento 2: Modelos Basados en Embeddings
- **GloVe**: Embeddings preentrenados con promediado
- **BERT**: Fine-tuning de modelo transformer
- **Objetivo**: Evaluar mejora con representaciones semánticas

### Experimento 3: Análisis Lingüístico
- **Interpretabilidad**: Análisis de características importantes
- **Patrones**: Identificación de elementos lingüísticos del clickbait
- **Objetivo**: Comprender qué hace clickbait a un titular

## 🚀 Instalación

### Requisitos
- Python 3.8+
- pip o uv (gestor de paquetes)

### Opción 1: Con UV (Recomendado)

```bash
# Instalar UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Instalar dependencias
uv sync
```

### Opción 2: Con pip

```bash
# Instalar dependencias
pip install pandas numpy scikit-learn matplotlib seaborn nltk tqdm

# Para Experimentos con embeddings (opcional)
pip install gensim transformers torch
```

## 📁 Estructura del Proyecto

```
tpe_nlp_2025/
├── main.py                      # Script principal (ejecuta todos los experimentos)
├── eda.py                       # Análisis exploratorio de datos
├── experimento1.py              # Experimento 1: Modelos clásicos
├── experimento2.py              # Experimento 2: Modelos con embeddings
├── experimento3.py              # Experimento 3: Análisis lingüístico
├── comparacion_final.py         # Comparación de todos los modelos
├── resumen_y_conclusiones.py    # Resumen y conclusiones del proyecto
├── assets/
│   └── clickbait_data.csv       # Dataset
├── pyproject.toml               # Dependencias del proyecto
└── README.md                    # Este archivo
```

## 🏃 Ejecución

### Ejecutar Análisis Completo

```bash
python main.py
```

Este script ejecutará en orden:
1. Análisis Exploratorio de Datos (EDA)
2. Experimento 1: Modelos Clásicos
3. Experimento 2: Modelos con Embeddings
4. Experimento 3: Análisis Lingüístico
5. Comparación Final de Modelos
6. Resumen y Conclusiones

### Ejecutar Experimentos Individuales

```bash
# Análisis exploratorio
python eda.py

# Experimento 1
python experimento1.py

# Experimento 2
python experimento2.py

# Experimento 3
python experimento3.py

# Comparación final
python comparacion_final.py

# Resumen y conclusiones
python resumen_y_conclusiones.py
```

## 📈 Resultados

Los experimentos generan:

### Visualizaciones (PNG)
- Gráficos de distribución de clases
- Comparaciones de modelos
- Análisis de características importantes
- Visualizaciones de patrones lingüísticos

### Datos (CSV)
- Resultados de cada experimento
- Tablas comparativas
- Características importantes
- Patrones lingüísticos identificados

### Documentación
- Resumen ejecutivo
- Conclusiones
- Limitaciones del estudio
- Trabajo futuro

## 📊 Dataset

El proyecto utiliza un dataset de aproximadamente 32,000 titulares de noticias en inglés, categorizados como:
- **Clickbait (1)**: Titulares engañosos diseñados para inducir clics
- **No Clickbait (0)**: Titulares informativos o neutrales

### Características del Dataset:
- Distribución balanceada (50% por clase)
- Longitud promedio: 5-13 palabras por titular
- Fuentes variadas de noticias

## 🔍 Preprocesamiento

1. **Limpieza**: Conversión a minúsculas, eliminación de caracteres especiales
2. **Tokenización**: División en palabras individuales
3. **Stopwords**: Eliminación de palabras comunes
4. **Lemmatización**: Reducción de palabras a su raíz

## 📏 Métricas de Evaluación

- **Accuracy**: Precisión general del modelo
- **Precision**: Proporción de verdaderos positivos entre todos los positivos
- **Recall**: Proporción de verdaderos positivos detectados
- **F1-Score**: Media armónica entre precision y recall

## 📝 Documentación Adicional

- `EXPERIMENTO2_README.md`: Documentación detallada del Experimento 2
- `EXPERIMENTO3_README.md`: Documentación detallada del Experimento 3

## 🤝 Contribuciones

Este es un proyecto académico. Para sugerencias o mejoras, por favor crea un issue.

## 📄 Licencia

Este proyecto es de uso académico.

## 👥 Autores

- Alberto Bendayan
- Theo Shlamovitz
- Theo Stanfield

## 📚 Referencias

- Dataset: [Clickbait Dataset - Kaggle](https://www.kaggle.com/datasets/amananandrai/clickbait-dataset)
- GloVe: [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- BERT: [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)

