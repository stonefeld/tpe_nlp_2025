# Análisis Exploratorio de Datos (EDA) para Clickbait Dataset
import re
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from nltk import bigrams, word_tokenize
from nltk.corpus import stopwords

# Descargar recursos NLTK
nltk.download("stopwords", raise_on_error=True)
nltk.download("punkt", raise_on_error=True)

# Configurar estilo
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def load_data():
    """Cargar dataset"""
    print("📂 Cargando dataset...")
    df = pd.read_csv("assets/clickbait_data.csv")
    df = df.dropna(subset=["headline", "clickbait"])
    print(f"✅ Dataset cargado: {df.shape[0]} titulares")
    return df


def analyze_class_distribution(df):
    """Análisis de distribución de clases"""
    print("\n" + "=" * 70)
    print("1. DISTRIBUCIÓN DE CLASES")
    print("=" * 70)
    
    class_counts = df["clickbait"].value_counts()
    print(f"\nDistribución:")
    print(f"  Clickbait (1):     {class_counts.get(1, 0):,} ({class_counts.get(1, 0)/len(df)*100:.1f}%)")
    print(f"  No Clickbait (0):  {class_counts.get(0, 0):,} ({class_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"  Total:             {len(df):,}")
    
    # Visualización
    fig, ax = plt.subplots(figsize=(10, 6))
    class_counts.plot(kind="bar", ax=ax, color=["steelblue", "crimson"], alpha=0.7)
    ax.set_title("Distribución de Clases", fontsize=14, fontweight="bold")
    ax.set_xlabel("Clase (0 = No Clickbait, 1 = Clickbait)", fontsize=12)
    ax.set_ylabel("Cantidad de Titulares", fontsize=12)
    ax.set_xticklabels(["No Clickbait", "Clickbait"], rotation=0)
    ax.grid(True, alpha=0.3, axis="y")
    
    for i, v in enumerate(class_counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig("eda_distribucion_clases.png", dpi=300, bbox_inches="tight")
    plt.show()


def analyze_headline_length(df):
    """Análisis de longitud de titulares"""
    print("\n" + "=" * 70)
    print("2. ANÁLISIS DE LONGITUD DE TITULARES")
    print("=" * 70)
    
    df["num_words"] = df["headline"].apply(lambda x: len(str(x).split()))
    df["num_chars"] = df["headline"].apply(lambda x: len(str(x)))
    
    # Estadísticas por clase
    stats = df.groupby("clickbait")[["num_words", "num_chars"]].describe()
    print("\n📊 Estadísticas descriptivas:")
    print(stats)
    
    # Visualización
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Histograma de palabras
    axes[0, 0].hist(
        df[df["clickbait"] == 0]["num_words"],
        bins=30,
        alpha=0.6,
        label="No Clickbait",
        color="steelblue",
        density=True,
    )
    axes[0, 0].hist(
        df[df["clickbait"] == 1]["num_words"],
        bins=30,
        alpha=0.6,
        label="Clickbait",
        color="crimson",
        density=True,
    )
    axes[0, 0].set_xlabel("Número de palabras", fontsize=11)
    axes[0, 0].set_ylabel("Densidad", fontsize=11)
    axes[0, 0].set_title("Distribución de Longitud (Palabras)", fontsize=12, fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Boxplot de palabras
    df_box = df[["num_words", "clickbait"]].copy()
    df_box["clickbait"] = df_box["clickbait"].map({0: "No Clickbait", 1: "Clickbait"})
    sns.boxplot(data=df_box, x="clickbait", y="num_words", ax=axes[0, 1], palette=["steelblue", "crimson"])
    axes[0, 1].set_title("Boxplot de Longitud (Palabras)", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("Número de palabras", fontsize=11)
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    
    # Histograma de caracteres
    axes[1, 0].hist(
        df[df["clickbait"] == 0]["num_chars"],
        bins=30,
        alpha=0.6,
        label="No Clickbait",
        color="steelblue",
        density=True,
    )
    axes[1, 0].hist(
        df[df["clickbait"] == 1]["num_chars"],
        bins=30,
        alpha=0.6,
        label="Clickbait",
        color="crimson",
        density=True,
    )
    axes[1, 0].set_xlabel("Número de caracteres", fontsize=11)
    axes[1, 0].set_ylabel("Densidad", fontsize=11)
    axes[1, 0].set_title("Distribución de Longitud (Caracteres)", fontsize=12, fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Comparación de promedios
    avg_words = df.groupby("clickbait")["num_words"].mean()
    avg_chars = df.groupby("clickbait")["num_chars"].mean()
    
    x = ["No Clickbait", "Clickbait"]
    width = 0.35
    x_pos = [0, 1]
    
    axes[1, 1].bar([p - width/2 for p in x_pos], [avg_words[0], avg_words[1]], width, label="Palabras", color="steelblue", alpha=0.7)
    axes[1, 1].bar([p + width/2 for p in x_pos], [avg_chars[0]/10, avg_chars[1]/10], width, label="Caracteres/10", color="crimson", alpha=0.7)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(x)
    axes[1, 1].set_ylabel("Valor Promedio", fontsize=11)
    axes[1, 1].set_title("Comparación de Promedios", fontsize=12, fontweight="bold")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig("eda_longitud_titulares.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    return df


def analyze_frequent_words(df):
    """Análisis de palabras más frecuentes"""
    print("\n" + "=" * 70)
    print("3. ANÁLISIS DE PALABRAS MÁS FRECUENTES")
    print("=" * 70)
    
    stop_words = set(stopwords.words("english"))
    
    def preprocess(text):
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", "", text)
        tokens = word_tokenize(text)
        return [w for w in tokens if w.isalpha() and w not in stop_words]
    
    df["tokens"] = df["headline"].apply(preprocess)
    
    def top_words(data, n=20):
        all_words = [w for tokens in data for w in tokens]
        freq = Counter(all_words).most_common(n)
        return pd.DataFrame(freq, columns=["word", "count"])
    
    top_clickbait = top_words(df[df["clickbait"] == 1]["tokens"])
    top_nonclickbait = top_words(df[df["clickbait"] == 0]["tokens"])
    
    print("\n📊 Top 20 palabras - CLICKBAIT:")
    print(top_clickbait.to_string(index=False))
    
    print("\n📊 Top 20 palabras - NO CLICKBAIT:")
    print(top_nonclickbait.to_string(index=False))
    
    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    
    # Palabras clickbait
    sns.barplot(y="word", x="count", data=top_clickbait, ax=axes[0], palette="Reds_r")
    axes[0].set_title("Palabras más frecuentes (Clickbait)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Frecuencia", fontsize=11)
    axes[0].set_ylabel("Palabras", fontsize=11)
    axes[0].grid(True, alpha=0.3, axis="x")
    
    # Palabras no clickbait
    sns.barplot(y="word", x="count", data=top_nonclickbait, ax=axes[1], palette="Blues_r")
    axes[1].set_title("Palabras más frecuentes (No Clickbait)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Frecuencia", fontsize=11)
    axes[1].set_ylabel("Palabras", fontsize=11)
    axes[1].grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    plt.savefig("eda_palabras_frecuentes.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    return top_clickbait, top_nonclickbait


def analyze_bigrams(df):
    """Análisis de bigramas más frecuentes"""
    print("\n" + "=" * 70)
    print("4. ANÁLISIS DE BIGRAMAS MÁS FRECUENTES")
    print("=" * 70)
    
    stop_words = set(stopwords.words("english"))
    
    def preprocess(text):
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", "", text)
        tokens = word_tokenize(text)
        return [w for w in tokens if w.isalpha() and w not in stop_words]
    
    def top_bigrams(data, n=15):
        all_bigrams = [bg for tokens in data for bg in bigrams(tokens)]
        freq = Counter(all_bigrams).most_common(n)
        return pd.DataFrame(freq, columns=["bigram", "count"])
    
    top_bigram_clickbait = top_bigrams(df[df["clickbait"] == 1]["tokens"])
    top_bigram_nonclickbait = top_bigrams(df[df["clickbait"] == 0]["tokens"])
    
    # Preparar datos para visualización
    top_bigram_clickbait["bigram_str"] = top_bigram_clickbait["bigram"].apply(lambda x: " ".join(x))
    top_bigram_nonclickbait["bigram_str"] = top_bigram_nonclickbait["bigram"].apply(lambda x: " ".join(x))
    
    print("\n📊 Top 15 bigramas - CLICKBAIT:")
    for idx, row in top_bigram_clickbait.iterrows():
        print(f"  {row['bigram_str']}: {row['count']}")
    
    print("\n📊 Top 15 bigramas - NO CLICKBAIT:")
    for idx, row in top_bigram_nonclickbait.iterrows():
        print(f"  {row['bigram_str']}: {row['count']}")
    
    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    
    sns.barplot(y="bigram_str", x="count", data=top_bigram_clickbait, ax=axes[0], palette="Oranges_r")
    axes[0].set_title("Bigramas más frecuentes (Clickbait)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Frecuencia", fontsize=11)
    axes[0].set_ylabel("Bigramas", fontsize=11)
    axes[0].grid(True, alpha=0.3, axis="x")
    
    sns.barplot(y="bigram_str", x="count", data=top_bigram_nonclickbait, ax=axes[1], palette="Greens_r")
    axes[1].set_title("Bigramas más frecuentes (No Clickbait)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Frecuencia", fontsize=11)
    axes[1].set_ylabel("Bigramas", fontsize=11)
    axes[1].grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    plt.savefig("eda_bigramas_frecuentes.png", dpi=300, bbox_inches="tight")
    plt.show()


def generate_summary_table(df):
    """Generar tabla resumen"""
    print("\n" + "=" * 70)
    print("5. TABLA RESUMEN")
    print("=" * 70)
    
    summary = pd.DataFrame({
        "Clase": ["No Clickbait", "Clickbait"],
        "Promedio palabras": df.groupby("clickbait")["num_words"].mean().values,
        "Titulares": df["clickbait"].value_counts().values,
    })
    
    print("\n📊 Tabla Resumen:")
    print(summary.to_string(index=False))
    
    # Visualización
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Promedio de palabras
    sns.barplot(data=summary, x="Clase", y="Promedio palabras", ax=axes[0], palette="viridis")
    axes[0].set_title("Promedio de palabras por clase", fontweight="bold")
    axes[0].set_ylabel("Promedio palabras")
    axes[0].grid(True, alpha=0.3, axis="y")
    
    # Distribución de titulares
    sns.barplot(data=summary, x="Clase", y="Titulares", ax=axes[1], palette="Set2")
    axes[1].set_title("Cantidad de titulares por clase", fontweight="bold")
    axes[1].set_ylabel("Cantidad de titulares")
    axes[1].grid(True, alpha=0.3, axis="y")
    
    # Comparación normalizada
    summary_norm = summary.copy()
    summary_norm["Promedio palabras"] = summary_norm["Promedio palabras"] / summary_norm["Promedio palabras"].max()
    summary_norm["Titulares"] = summary_norm["Titulares"] / summary_norm["Titulares"].max()
    
    summary_melted = summary_norm.melt(id_vars=["Clase"], var_name="Métrica", value_name="Valor Normalizado")
    sns.barplot(data=summary_melted, x="Métrica", y="Valor Normalizado", hue="Clase", ax=axes[2], palette="Set1")
    axes[2].set_title("Comparación de métricas (normalizadas)", fontweight="bold")
    axes[2].set_ylabel("Valor Normalizado")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig("eda_tabla_resumen.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # Guardar resumen
    summary.to_csv("eda_resumen.csv", index=False)
    print("\n✅ Resumen guardado en: eda_resumen.csv")


def main():
    """Función principal"""
    print("=" * 70)
    print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("=" * 70)
    
    # Cargar datos
    df = load_data()
    
    # Análisis de distribución
    analyze_class_distribution(df)
    
    # Análisis de longitud
    df = analyze_headline_length(df)
    
    # Análisis de palabras frecuentes
    analyze_frequent_words(df)
    
    # Análisis de bigramas
    analyze_bigrams(df)
    
    # Tabla resumen
    generate_summary_table(df)
    
    print("\n" + "=" * 70)
    print("✅ EDA COMPLETADO")
    print("=" * 70)
    print("\n📁 Archivos generados:")
    print("   • eda_distribucion_clases.png")
    print("   • eda_longitud_titulares.png")
    print("   • eda_palabras_frecuentes.png")
    print("   • eda_bigramas_frecuentes.png")
    print("   • eda_tabla_resumen.png")
    print("   • eda_resumen.csv")


if __name__ == "__main__":
    main()

