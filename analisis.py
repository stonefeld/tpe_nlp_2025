# Exploratory Data Analysis for Clickbait Dataset
# Dataset: https://www.kaggle.com/datasets/amananandrai/clickbait-dataset

from collections import Counter

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from nltk import bigrams, word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# === 1. Load dataset ===
df = pd.read_csv("assets/clickbait_data.csv")  # or adjust filename

# Preview data
print(df.head())
print(df["clickbait"].value_counts())

# === 2. Distribution of classes ===
sns.countplot(x="clickbait", data=df)
plt.title("Distribución de clases (0 = No clickbait, 1 = Clickbait)")
plt.xlabel("Clase")
plt.ylabel("Cantidad de titulares")
plt.show()

# === 3. Compute headline lengths ===
df["num_words"] = df["headline"].apply(lambda x: len(str(x).split()))

# Summary statistics by class
stats = df.groupby("clickbait")["num_words"].describe()
print(stats)

# Visualize length distributions
fig, ax = plt.subplots(figsize=(15, 6))

# Histograma de palabras
sns.histplot(data=df, x="num_words", hue="clickbait", bins=20, kde=True, ax=ax)
ax.set_title("Distribución de longitud de titulares (en palabras)", fontweight='bold')
ax.set_xlabel("Cantidad de palabras")
ax.set_ylabel("Frecuencia")

plt.tight_layout()
plt.show()

# === 4. Preprocessing for text analysis ===
stop_words = set(stopwords.words("english"))


def preprocess(text):
    tokens = word_tokenize(text.lower())
    return [w for w in tokens if w.isalpha() and (w not in stop_words)]


df["tokens"] = df["headline"].apply(preprocess)


# === 5. Most frequent words per class ===
def top_words(data, n=20):
    all_words = [w for tokens in data for w in tokens]
    freq = Counter(all_words).most_common(n)
    return pd.DataFrame(freq, columns=["word", "count"])


top_clickbait = top_words(df[df["clickbait"] == 1]["tokens"])
top_nonclickbait = top_words(df[df["clickbait"] == 0]["tokens"])

# Plot palabras más frecuentes
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
sns.barplot(y="word", x="count", data=top_clickbait, ax=axes[0], palette="Reds_r")
axes[0].set_title("Palabras más frecuentes (Clickbait)", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Frecuencia")
axes[0].set_ylabel("Palabras")

sns.barplot(y="word", x="count", data=top_nonclickbait, ax=axes[1], palette="Blues_r")
axes[1].set_title("Palabras más frecuentes (No Clickbait)", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Frecuencia")
axes[1].set_ylabel("Palabras")

plt.tight_layout()
plt.show()


# === 6. Bigram analysis ===
def top_bigrams(data, n=15):
    all_bigrams = [bg for tokens in data for bg in bigrams(tokens)]
    freq = Counter(all_bigrams).most_common(n)
    return pd.DataFrame(freq, columns=["bigram", "count"])


top_bigram_clickbait = top_bigrams(df[df["clickbait"] == 1]["tokens"])
top_bigram_nonclickbait = top_bigrams(df[df["clickbait"] == 0]["tokens"])

# Plot bigramas más frecuentes
fig, axes = plt.subplots(1, 2, figsize=(18, 10))

# Preparar datos para bigramas (convertir tuplas a strings)
top_bigram_clickbait['bigram_str'] = top_bigram_clickbait['bigram'].apply(lambda x: ' '.join(x))
top_bigram_nonclickbait['bigram_str'] = top_bigram_nonclickbait['bigram'].apply(lambda x: ' '.join(x))

sns.barplot(y="bigram_str", x="count", data=top_bigram_clickbait, ax=axes[0], palette="Oranges_r")
axes[0].set_title("Bigramas más frecuentes (Clickbait)", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Frecuencia")
axes[0].set_ylabel("Bigramas")

sns.barplot(y="bigram_str", x="count", data=top_bigram_nonclickbait, ax=axes[1], palette="Greens_r")
axes[1].set_title("Bigramas más frecuentes (No Clickbait)", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Frecuencia")
axes[1].set_ylabel("Bigramas")

plt.tight_layout()
plt.show()

# === 7. Summary table as plots ===
summary = pd.DataFrame(
    {
        "Clase": ["No Clickbait", "Clickbait"],
        "Promedio palabras": df.groupby("clickbait")["num_words"].mean().values,
        "Titulares": df["clickbait"].value_counts().values,
    }
)

# Crear visualizaciones para la tabla resumen
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. Promedio de palabras por clase
sns.barplot(data=summary, x="Clase", y="Promedio palabras", ax=axes[0], palette="viridis")
axes[0].set_title("Promedio de palabras por clase", fontweight='bold')
axes[0].set_ylabel("Promedio palabras")

# 2. Distribución de titulares
sns.barplot(data=summary, x="Clase", y="Titulares", ax=axes[1], palette="Set2")
axes[1].set_title("Cantidad de titulares por clase", fontweight='bold')
axes[1].set_ylabel("Cantidad de titulares")

# 3. Comparación de métricas (normalizadas)
summary_norm = summary.copy()
summary_norm["Promedio palabras"] = summary_norm["Promedio palabras"] / summary_norm["Promedio palabras"].max()
summary_norm["Titulares"] = summary_norm["Titulares"] / summary_norm["Titulares"].max()

summary_melted = summary_norm.melt(id_vars=['Clase'], var_name='Métrica', value_name='Valor Normalizado')
sns.barplot(data=summary_melted, x="Métrica", y="Valor Normalizado", hue="Clase", ax=axes[2], palette="Set1")
axes[2].set_title("Comparación de métricas (normalizadas)", fontweight='bold')
axes[2].set_ylabel("Valor Normalizado")
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# También imprimir la tabla para referencia
print("\n=== TABLA RESUMEN ===")
print(summary)
