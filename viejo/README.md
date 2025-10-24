# Detección Automática de Titulares Clickbait en Noticias Digitales

## Descripción del Proyecto

Este proyecto implementa un sistema de clasificación automática para detectar titulares clickbait en noticias digitales utilizando técnicas de procesamiento de lenguaje natural (NLP). El objetivo es distinguir entre titulares engañosos diseñados para atraer clics y titulares informativos o neutrales.

## Objetivos

1. **Clasificación binaria**: Entrenar modelos para clasificar titulares como clickbait o no clickbait
2. **Análisis comparativo**: Evaluar diferentes representaciones de texto (BoW, TF-IDF, embeddings)
3. **Interpretabilidad**: Identificar patrones lingüísticos característicos del clickbait

## Metodología

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

## Estructura del Proyecto

```
tpe_nlp_2025/
├── main.py                     # Script principal
├── clickbait_detection.py      # Clase principal del detector
├── download_dataset.py         # Script para descargar datos
├── clickbait_analysis.ipynb    # Notebook de análisis interactivo
├── requirements.txt            # Dependencias del proyecto
├── README.md                   # Este archivo
└── data/                       # Directorio de datos
    └── clickbait_sample.csv    # Dataset de muestra
```

## Instalación y Uso

### Opción 1: Con UV (Recomendado)

#### 1. Instalar UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. Configuración Automática
```bash
python setup_uv.py
```

#### 3. Ejecutar Análisis
```bash
uv run python main.py
```

### Opción 2: Con pip (Tradicional)

#### 1. Instalar Dependencias
```bash
pip install -r requirements.txt
```

#### 2. Ejecutar Análisis Completo
```bash
python main.py
```

### Análisis Interactivo (Jupyter)

```bash
# Con UV
uv run jupyter notebook clickbait_analysis.ipynb

# Con pip
jupyter notebook clickbait_analysis.ipynb
```

## Dataset

El proyecto utiliza un dataset de aproximadamente 32,000 titulares de noticias en inglés, categorizados como:
- **Clickbait (1)**: Titulares engañosos diseñados para inducir clics
- **No Clickbait (0)**: Titulares informativos o neutrales

### Características del Dataset:
- Distribución balanceada (50% por clase)
- Longitud promedio: 5-13 palabras por titular
- Fuentes confiables vs. no confiables

## Preprocesamiento

1. **Limpieza**: Conversión a minúsculas, eliminación de caracteres especiales
2. **Tokenización**: División en palabras individuales
3. **Stopwords**: Eliminación de palabras comunes (the, is, in, etc.)
4. **Stemming**: Reducción de palabras a su raíz (opcional)

## Métricas de Evaluación

- **Accuracy**: Precisión general del modelo
- **Precision**: Proporción de verdaderos positivos entre todos los positivos
- **Recall**: Proporción de verdaderos positivos detectados
- **F1-Score**: Media armónica entre precision y recall

## Resultados Esperados

### Patrones de Clickbait Identificados:
- Uso de pronombres en segunda persona (you, your)
- Verbos imperativos y de acción (won't, believe, happen)
- Palabras de intensidad emocional (shock, amazing, secret)
- Frases incompletas o suspensivas (...)
- Números y listas (10 things, one trick)
- Palabras de urgencia (next, forever, never)

### Patrones de No Clickbait:
- Lenguaje informativo y directo
- Referencias a entidades específicas (scientists, president, company)
- Verbos en tiempo presente o pasado simple
- Información factual y verificable
- Ausencia de elementos persuasivos

## Archivos de Salida

El proyecto genera varios archivos de salida:
- `class_distribution.png`: Distribución de clases
- `headline_length_analysis.png`: Análisis de longitud de titulares
- `confusion_matrix_*.png`: Matrices de confusión por modelo
- `linguistic_analysis.png`: Análisis de características lingüísticas

## Dependencias

- pandas==2.0.3
- numpy==1.24.3
- scikit-learn==1.3.0
- nltk==3.8.1
- matplotlib==3.7.2
- seaborn==0.12.2
- wordcloud==1.9.2
- transformers==4.33.2
- torch==2.0.1
- datasets==2.14.4
- kaggle==1.5.16
- jupyter==1.0.0

## Configuración de Kaggle (Opcional)

Para usar el dataset real de Kaggle:

1. Ve a https://www.kaggle.com/account
2. Crea un nuevo token API
3. Descarga kaggle.json
4. Colócalo en ~/.kaggle/kaggle.json
5. Ejecuta: `chmod 600 ~/.kaggle/kaggle.json`

## Contribuciones

Este proyecto es parte de un trabajo práctico de NLP. Para contribuciones o mejoras, por favor contacta al autor.

## Licencia

Este proyecto es de uso académico y educativo.
