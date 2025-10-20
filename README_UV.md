# Detección Automática de Titulares Clickbait - Configuración con UV

## ¿Qué es UV?

[UV](https://github.com/astral-sh/uv) es un gestor de paquetes y entornos virtuales de Python extremadamente rápido, escrito en Rust. Es una alternativa moderna a pip y pip-tools que ofrece:

- ⚡ **Velocidad**: 10-100x más rápido que pip
- 🔒 **Reproducibilidad**: Lock files determinísticos
- 🎯 **Simplicidad**: Comando único para instalar dependencias
- 🛡️ **Seguridad**: Resolución de dependencias más robusta

## Instalación de UV

### Opción 1: Instalador oficial (Recomendado)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Opción 2: Con pip
```bash
pip install uv
```

### Opción 3: Con conda
```bash
conda install -c conda-forge uv
```

## Configuración del Proyecto

### 1. Configuración Automática
```bash
python setup_uv.py
```

### 2. Configuración Manual

#### Instalar dependencias básicas:
```bash
uv sync
```

#### Instalar dependencias opcionales:
```bash
# Para experimentos con embeddings (Word2Vec, BERT, etc.)
uv sync --extra embeddings

# Para descargar datasets de Kaggle
uv sync --extra kaggle

# Para desarrollo (testing, linting, etc.)
uv sync --extra dev

# Instalar todo
uv sync --all-extras
```

## Uso del Proyecto con UV

### Comandos Principales

#### Análisis Completo
```bash
uv run python main.py
```

#### Demostración Interactiva
```bash
uv run python demo.py
```

#### Experimentos de Embeddings
```bash
uv run python embeddings_experiment.py
```

#### Análisis en Jupyter
```bash
uv run jupyter notebook clickbait_analysis.ipynb
```

### Scripts de Conveniencia

Después de ejecutar `setup_uv.py`, tendrás scripts de conveniencia:

```bash
python run_analysis.py      # Análisis completo
python run_demo.py          # Demostración
python run_embeddings.py    # Experimentos de embeddings
```

## Estructura de Dependencias

### Dependencias Básicas (siempre instaladas)
- `pandas` - Manipulación de datos
- `numpy` - Computación numérica
- `scikit-learn` - Machine learning
- `nltk` - Procesamiento de lenguaje natural
- `matplotlib` - Visualizaciones
- `seaborn` - Visualizaciones estadísticas
- `wordcloud` - Nubes de palabras
- `jupyter` - Notebooks interactivos
- `tqdm` - Barras de progreso

### Dependencias Opcionales

#### `embeddings` - Para experimentos avanzados
- `transformers` - Modelos BERT, etc.
- `torch` - PyTorch para deep learning
- `datasets` - Datasets de Hugging Face
- `gensim` - Word2Vec, GloVe, etc.

#### `kaggle` - Para descargar datasets
- `kaggle` - API de Kaggle

#### `dev` - Para desarrollo
- `pytest` - Testing
- `black` - Formateo de código
- `flake8` - Linting
- `mypy` - Type checking

## Ventajas de usar UV

### 1. Velocidad
```bash
# Instalación típica con pip: ~30-60 segundos
# Instalación con uv: ~3-5 segundos
uv sync
```

### 2. Reproducibilidad
```bash
# Lock file determinístico
uv lock

# Instalación exacta de versiones
uv sync --frozen
```

### 3. Gestión de Entornos
```bash
# Crear entorno virtual automáticamente
uv run python script.py

# Activar entorno manualmente
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### 4. Resolución de Dependencias
```bash
# Verificar dependencias
uv tree

# Actualizar dependencias
uv lock --upgrade
```

## Migración desde pip

Si ya tienes un `requirements.txt`:

```bash
# Convertir a pyproject.toml
uv init --app

# Instalar desde requirements.txt
uv pip install -r requirements.txt

# Generar lock file
uv lock
```

## Comandos Útiles

### Gestión de Dependencias
```bash
uv add pandas              # Agregar dependencia
uv add --dev pytest       # Agregar dependencia de desarrollo
uv remove pandas          # Remover dependencia
uv sync                   # Instalar todas las dependencias
uv lock                   # Generar/actualizar lock file
```

### Ejecución
```bash
uv run python script.py   # Ejecutar script
uv run jupyter notebook   # Ejecutar Jupyter
uv run pytest            # Ejecutar tests
```

### Información
```bash
uv tree                   # Ver árbol de dependencias
uv show                   # Mostrar información del proyecto
uv --version              # Versión de uv
```

## Troubleshooting

### Error: "uv: command not found"
```bash
# Reinstalar uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# O agregar al PATH
export PATH="$HOME/.cargo/bin:$PATH"
```

### Error: "No module named 'clickbait_detection'"
```bash
# Asegúrate de estar en el directorio del proyecto
cd /path/to/tpe_nlp_2025

# Ejecutar con uv
uv run python main.py
```

### Error: "Package not found"
```bash
# Verificar que la dependencia esté en pyproject.toml
uv add missing-package

# O instalar manualmente
uv add --index-url https://pypi.org/simple/ package-name
```

## Comparación: UV vs Pip

| Característica | UV | Pip |
|----------------|----|----|
| Velocidad | ⚡ 10-100x más rápido | 🐌 Lento |
| Lock files | ✅ Determinísticos | ❌ No nativo |
| Resolución | 🎯 Robusta | ⚠️ Básica |
| Entornos | 🔄 Automático | 🔧 Manual |
| Instalación | 📦 Un comando | 🔧 Múltiples pasos |

## Recursos Adicionales

- [Documentación oficial de UV](https://github.com/astral-sh/uv)
- [Guía de migración desde pip](https://github.com/astral-sh/uv/blob/main/README.md#migrating-from-pip)
- [Comandos de UV](https://github.com/astral-sh/uv/blob/main/README.md#commands)
