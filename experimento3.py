# EXPERIMENTO 3: Análisis Lingüístico de Patrones de Clickbait
import re
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk import bigrams, pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm import tqdm

# Descargar recursos NLTK
nltk.download("stopwords", raise_on_error=True)
nltk.download("wordnet", raise_on_error=True)
nltk.download("omw-1.4", raise_on_error=True)
nltk.download("averaged_perceptron_tagger_eng", raise_on_error=True)
nltk.download("punkt", raise_on_error=True)

# ============================================
# FUNCIONES DE PREPROCESAMIENTO (iguales al Exp1)
# ============================================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    lemmas = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged]
    return " ".join(lemmas)


# ============================================
# FUNCIONES PARA ANÁLISIS LINGÜÍSTICO
# ============================================

def analyze_linguistic_patterns(texts, labels):
    """
    Analizar patrones lingüísticos específicos en los textos
    """
    patterns = {
        "question_words": ["what", "how", "why", "when", "where", "which", "who", "whom", "whose"],
        "imperative_words": ["must", "should", "need", "have to", "got to", "gotta", "will", "won't"],
        "emotional_words": [
            "amazing", "shocking", "incredible", "unbelievable", "stunning", "wow", "mind-blowing",
            "you won't believe", "secret", "hidden", "revealed", "shocking truth", "exposed",
        ],
        "number_words": ["one", "two", "three", "four", "five", "ten", "hundred", "thousand", "million"],
        "personal_pronouns": ["you", "your", "yours", "yourself", "yourselves"],
        "urgency_words": ["now", "today", "this instant", "immediately", "urgent", "breaking"],
        "superlatives": ["best", "worst", "biggest", "smallest", "most", "least", "first", "last"],
    }

    results = {}

    for pattern_name, pattern_words in patterns.items():
        clickbait_count = 0
        non_clickbait_count = 0
        clickbait_examples = []
        non_clickbait_examples = []

        for text, label in zip(texts, labels):
            text_lower = str(text).lower()
            # Verificar si alguna palabra/frase del patrón está en el texto
            has_pattern = False
            for pattern in pattern_words:
                # Para frases (con espacios), buscar la frase completa
                if " " in pattern:
                    if pattern in text_lower:
                        has_pattern = True
                        break
                else:
                    # Para palabras, buscar la palabra en el texto
                    # Usar regex para buscar como palabra completa (evitar subcadenas)
                    if re.search(r'\b' + re.escape(pattern) + r'\b', text_lower):
                        has_pattern = True
                        break

            if has_pattern:
                if label == 1:
                    clickbait_count += 1
                    if len(clickbait_examples) < 3:
                        clickbait_examples.append(text)
                else:
                    non_clickbait_count += 1
                    if len(non_clickbait_examples) < 3:
                        non_clickbait_examples.append(text)

        total = clickbait_count + non_clickbait_count
        results[pattern_name] = {
            "clickbait": clickbait_count,
            "non_clickbait": non_clickbait_count,
            "total": total,
            "clickbait_examples": clickbait_examples,
            "non_clickbait_examples": non_clickbait_examples,
        }

    return results


def get_top_words_by_class(texts, labels, n=20):
    """
    Obtener palabras más frecuentes por clase
    """
    clickbait_texts = [str(text).lower() for text, label in zip(texts, labels) if label == 1]
    non_clickbait_texts = [str(text).lower() for text, label in zip(texts, labels) if label == 0]

    def tokenize_and_count(text_list):
        all_words = []
        for text in text_list:
            words = word_tokenize(text)
            words = [w for w in words if w.isalpha() and w not in stop_words]
            all_words.extend(words)
        return Counter(all_words)

    clickbait_counter = tokenize_and_count(clickbait_texts)
    non_clickbait_counter = tokenize_and_count(non_clickbait_texts)

    top_clickbait = clickbait_counter.most_common(n)
    top_non_clickbait = non_clickbait_counter.most_common(n)

    return top_clickbait, top_non_clickbait


def get_top_bigrams_by_class(texts, labels, n=15):
    """
    Obtener bigramas más frecuentes por clase
    """
    clickbait_texts = [str(text).lower() for text, label in zip(texts, labels) if label == 1]
    non_clickbait_texts = [str(text).lower() for text, label in zip(texts, labels) if label == 0]

    def get_bigrams(text_list):
        all_bigrams = []
        for text in text_list:
            words = word_tokenize(text)
            words = [w for w in words if w.isalpha() and w not in stop_words]
            bigram_list = list(bigrams(words))
            all_bigrams.extend(bigram_list)
        return Counter(all_bigrams)

    clickbait_bigrams = get_bigrams(clickbait_texts)
    non_clickbait_bigrams = get_bigrams(non_clickbait_texts)

    top_clickbait = clickbait_bigrams.most_common(n)
    top_non_clickbait = non_clickbait_bigrams.most_common(n)

    return top_clickbait, top_non_clickbait


# ============================================
# FUNCIÓN PRINCIPAL
# ============================================

def main():
    print("=" * 70)
    print("EXPERIMENTO 3: ANÁLISIS LINGÜÍSTICO DE PATRONES DE CLICKBAIT")
    print("=" * 70)

    # Cargar dataset
    print("\n📂 Cargando dataset...")
    df = pd.read_csv("assets/clickbait_data.csv")
    df = df.dropna(subset=["headline", "clickbait"])
    print(f"✅ Dataset cargado: {df.shape[0]} titulares")
    print(f"   Distribución: {df['clickbait'].value_counts().to_dict()}")

    # Preprocesar texto
    print("\n🔄 Preprocesando texto...")
    tqdm.pandas(desc="Preprocesando titulares")
    df["clean"] = df["headline"].progress_apply(clean_text)

    # Dividir datos (necesitamos entrenar el modelo primero)
    print("\n📊 Dividiendo datos...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        df["clean"], df["clickbait"], test_size=0.3, random_state=42, stratify=df["clickbait"]
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"   Train: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}")

    # ============================================
    # PARTE 1: ENTRENAR MODELO TF-IDF PARA ANÁLISIS
    # ============================================
    print("\n" + "=" * 70)
    print("PARTE 1: ENTRENANDO MODELO TF-IDF PARA ANÁLISIS")
    print("=" * 70)

    print("\n🔄 Entrenando modelo TF-IDF + Regresión Logística...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_valid_vec = vectorizer.transform(X_valid)

    # Entrenar modelo
    param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
    lr = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
    grid = GridSearchCV(lr, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=0)
    grid.fit(X_train_vec, y_train)

    print(f"✅ Modelo entrenado. Mejor C: {grid.best_params_['C']}")
    best_model = grid.best_estimator_

    # ============================================
    # PARTE 2: ANÁLISIS DE CARACTERÍSTICAS IMPORTANTES
    # ============================================
    print("\n" + "=" * 70)
    print("PARTE 2: ANÁLISIS DE CARACTERÍSTICAS IMPORTANTES")
    print("=" * 70)

    # Obtener características y coeficientes
    feature_names = vectorizer.get_feature_names_out()
    coefficients = best_model.coef_[0]

    # Crear DataFrame con importancia de características
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "coefficient": coefficients}
    ).sort_values("coefficient", ascending=False)

    # Top características que indican clickbait (coeficientes positivos)
    top_clickbait_features = feature_importance.head(20)
    print("\n📊 Top 20 características que indican CLICKBAIT:")
    print(top_clickbait_features.to_string(index=False))

    # Top características que indican no clickbait (coeficientes negativos)
    top_non_clickbait_features = feature_importance.tail(20).sort_values("coefficient")
    print("\n📊 Top 20 características que indican NO CLICKBAIT:")
    print(top_non_clickbait_features.to_string(index=False))

    # Visualización de características importantes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Gráfico de características clickbait
    top_clickbait_plot = top_clickbait_features.head(15)
    ax1.barh(range(len(top_clickbait_plot)), top_clickbait_plot["coefficient"], color="crimson", alpha=0.7)
    ax1.set_yticks(range(len(top_clickbait_plot)))
    ax1.set_yticklabels(top_clickbait_plot["feature"], fontsize=10)
    ax1.set_xlabel("Coeficiente", fontsize=12)
    ax1.set_title("Top 15 Características - CLICKBAIT", fontsize=14, fontweight="bold")
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis="x")

    # Gráfico de características no clickbait
    top_non_clickbait_plot = top_non_clickbait_features.head(15)
    ax2.barh(
        range(len(top_non_clickbait_plot)),
        top_non_clickbait_plot["coefficient"],
        color="steelblue",
        alpha=0.7,
    )
    ax2.set_yticks(range(len(top_non_clickbait_plot)))
    ax2.set_yticklabels(top_non_clickbait_plot["feature"], fontsize=10)
    ax2.set_xlabel("Coeficiente", fontsize=12)
    ax2.set_title("Top 15 Características - NO CLICKBAIT", fontsize=14, fontweight="bold")
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("experimento3_caracteristicas_importantes.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Guardar resultados
    feature_importance.to_csv("experimento3_caracteristicas_importantes.csv", index=False)
    print("\n✅ Características importantes guardadas en: experimento3_caracteristicas_importantes.csv")

    # ============================================
    # PARTE 3: ANÁLISIS DE PATRONES LINGÜÍSTICOS ESPECÍFICOS
    # ============================================
    print("\n" + "=" * 70)
    print("PARTE 3: ANÁLISIS DE PATRONES LINGÜÍSTICOS ESPECÍFICOS")
    print("=" * 70)

    # Analizar patrones en el dataset completo
    print("\n🔍 Analizando patrones lingüísticos en el dataset completo...")
    pattern_results = analyze_linguistic_patterns(df["headline"], df["clickbait"])

    print("\n📊 PATRONES LINGÜÍSTICOS ENCONTRADOS:")
    print("-" * 70)

    pattern_summary = []
    for pattern_name, counts in pattern_results.items():
        total = counts["total"]
        if total > 0:
            cb_ratio = counts["clickbait"] / total * 100
            ncb_ratio = counts["non_clickbait"] / total * 100
            pattern_pct = total / len(df) * 100

            pattern_summary.append(
                {
                    "Patrón": pattern_name.replace("_", " ").title(),
                    "Total": total,
                    "% Dataset": f"{pattern_pct:.1f}%",
                    "Clickbait": f"{counts['clickbait']} ({cb_ratio:.1f}%)",
                    "No Clickbait": f"{counts['non_clickbait']} ({ncb_ratio:.1f}%)",
                    "Ratio CB": f"{cb_ratio:.1f}%",
                }
            )

            print(f"\n{pattern_name.upper().replace('_', ' ')}:")
            print(f"  Total: {total:,} ({pattern_pct:.1f}% del dataset)")
            print(f"  Clickbait: {counts['clickbait']:,} ({cb_ratio:.1f}%)")
            print(f"  No Clickbait: {counts['non_clickbait']:,} ({ncb_ratio:.1f}%)")

            if counts["clickbait_examples"]:
                print(f"  Ejemplos Clickbait: {counts['clickbait_examples'][:2]}")

    # Guardar resumen de patrones
    df_patterns = pd.DataFrame(pattern_summary)
    df_patterns.to_csv("experimento3_patrones_linguisticos.csv", index=False)
    print("\n✅ Patrones lingüísticos guardados en: experimento3_patrones_linguisticos.csv")

    # Visualización de patrones
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 1. Distribución de patrones por clase
    pattern_names = [p.replace("_", " ").title() for p in pattern_results.keys()]
    clickbait_counts = [pattern_results[p]["clickbait"] for p in pattern_results.keys()]
    non_clickbait_counts = [pattern_results[p]["non_clickbait"] for p in pattern_results.keys()]

    x = np.arange(len(pattern_names))
    width = 0.35

    axes[0, 0].bar(x - width / 2, clickbait_counts, width, label="Clickbait", color="crimson", alpha=0.7)
    axes[0, 0].bar(x + width / 2, non_clickbait_counts, width, label="No Clickbait", color="steelblue", alpha=0.7)
    axes[0, 0].set_xlabel("Patrones", fontsize=11)
    axes[0, 0].set_ylabel("Cantidad", fontsize=11)
    axes[0, 0].set_title("Distribución de Patrones por Clase", fontsize=12, fontweight="bold")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(pattern_names, rotation=45, ha="right", fontsize=9)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # 2. Proporción de patrones por clase (normalizado)
    cb_ratios = [
        pattern_results[p]["clickbait"] / max(pattern_results[p]["total"], 1) * 100
        for p in pattern_results.keys()
    ]
    ncb_ratios = [
        pattern_results[p]["non_clickbait"] / max(pattern_results[p]["total"], 1) * 100
        for p in pattern_results.keys()
    ]

    axes[0, 1].bar(x - width / 2, cb_ratios, width, label="Clickbait", color="crimson", alpha=0.7)
    axes[0, 1].bar(x + width / 2, ncb_ratios, width, label="No Clickbait", color="steelblue", alpha=0.7)
    axes[0, 1].set_xlabel("Patrones", fontsize=11)
    axes[0, 1].set_ylabel("Porcentaje (%)", fontsize=11)
    axes[0, 1].set_title("Proporción de Patrones por Clase", fontsize=12, fontweight="bold")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(pattern_names, rotation=45, ha="right", fontsize=9)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # 3. Frecuencia total de patrones
    total_counts = [pattern_results[p]["total"] for p in pattern_results.keys()]
    axes[1, 0].bar(range(len(pattern_names)), total_counts, color="mediumseagreen", alpha=0.7)
    axes[1, 0].set_xlabel("Patrones", fontsize=11)
    axes[1, 0].set_ylabel("Frecuencia Total", fontsize=11)
    axes[1, 0].set_title("Frecuencia Total de Patrones", fontsize=12, fontweight="bold")
    axes[1, 0].set_xticks(range(len(pattern_names)))
    axes[1, 0].set_xticklabels(pattern_names, rotation=45, ha="right", fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # 4. Ratio Clickbait/No Clickbait por patrón
    ratios = [
        pattern_results[p]["clickbait"] / max(pattern_results[p]["non_clickbait"], 1)
        for p in pattern_results.keys()
    ]
    axes[1, 1].bar(range(len(pattern_names)), ratios, color="orange", alpha=0.7)
    axes[1, 1].axhline(y=1, color="black", linestyle="--", linewidth=1, label="Ratio 1:1")
    axes[1, 1].set_xlabel("Patrones", fontsize=11)
    axes[1, 1].set_ylabel("Ratio Clickbait / No Clickbait", fontsize=11)
    axes[1, 1].set_title("Ratio de Clickbait por Patrón", fontsize=12, fontweight="bold")
    axes[1, 1].set_xticks(range(len(pattern_names)))
    axes[1, 1].set_xticklabels(pattern_names, rotation=45, ha="right", fontsize=9)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("experimento3_patrones_linguisticos.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ============================================
    # PARTE 4: ANÁLISIS DE PALABRAS Y BIGRAMAS MÁS FRECUENTES
    # ============================================
    print("\n" + "=" * 70)
    print("PARTE 4: ANÁLISIS DE PALABRAS Y BIGRAMAS MÁS FRECUENTES")
    print("=" * 70)

    print("\n🔍 Analizando palabras más frecuentes por clase...")
    top_clickbait_words, top_non_clickbait_words = get_top_words_by_class(df["headline"], df["clickbait"], n=20)

    print("\n📊 Top 20 palabras más frecuentes - CLICKBAIT:")
    for word, count in top_clickbait_words:
        print(f"  {word}: {count}")

    print("\n📊 Top 20 palabras más frecuentes - NO CLICKBAIT:")
    for word, count in top_non_clickbait_words:
        print(f"  {word}: {count}")

    # Visualización de palabras más frecuentes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

    # Palabras clickbait
    words_cb = [w for w, _ in top_clickbait_words]
    counts_cb = [c for _, c in top_clickbait_words]
    ax1.barh(range(len(words_cb)), counts_cb, color="crimson", alpha=0.7)
    ax1.set_yticks(range(len(words_cb)))
    ax1.set_yticklabels(words_cb, fontsize=11)
    ax1.set_xlabel("Frecuencia", fontsize=12)
    ax1.set_title("Top 20 Palabras - CLICKBAIT", fontsize=14, fontweight="bold")
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis="x")

    # Palabras no clickbait
    words_ncb = [w for w, _ in top_non_clickbait_words]
    counts_ncb = [c for _, c in top_non_clickbait_words]
    ax2.barh(range(len(words_ncb)), counts_ncb, color="steelblue", alpha=0.7)
    ax2.set_yticks(range(len(words_ncb)))
    ax2.set_yticklabels(words_ncb, fontsize=11)
    ax2.set_xlabel("Frecuencia", fontsize=12)
    ax2.set_title("Top 20 Palabras - NO CLICKBAIT", fontsize=14, fontweight="bold")
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("experimento3_palabras_frecuentes.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Bigramas más frecuentes
    print("\n🔍 Analizando bigramas más frecuentes por clase...")
    top_clickbait_bigrams, top_non_clickbait_bigrams = get_top_bigrams_by_class(
        df["headline"], df["clickbait"], n=15
    )

    print("\n📊 Top 15 bigramas más frecuentes - CLICKBAIT:")
    for bigram, count in top_clickbait_bigrams:
        print(f"  {' '.join(bigram)}: {count}")

    print("\n📊 Top 15 bigramas más frecuentes - NO CLICKBAIT:")
    for bigram, count in top_non_clickbait_bigrams:
        print(f"  {' '.join(bigram)}: {count}")

    # Visualización de bigramas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

    # Bigramas clickbait
    bigrams_cb = [" ".join(bg) for bg, _ in top_clickbait_bigrams]
    counts_bg_cb = [c for _, c in top_clickbait_bigrams]
    ax1.barh(range(len(bigrams_cb)), counts_bg_cb, color="crimson", alpha=0.7)
    ax1.set_yticks(range(len(bigrams_cb)))
    ax1.set_yticklabels(bigrams_cb, fontsize=11)
    ax1.set_xlabel("Frecuencia", fontsize=12)
    ax1.set_title("Top 15 Bigramas - CLICKBAIT", fontsize=14, fontweight="bold")
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis="x")

    # Bigramas no clickbait
    bigrams_ncb = [" ".join(bg) for bg, _ in top_non_clickbait_bigrams]
    counts_bg_ncb = [c for _, c in top_non_clickbait_bigrams]
    ax2.barh(range(len(bigrams_ncb)), counts_bg_ncb, color="steelblue", alpha=0.7)
    ax2.set_yticks(range(len(bigrams_ncb)))
    ax2.set_yticklabels(bigrams_ncb, fontsize=11)
    ax2.set_xlabel("Frecuencia", fontsize=12)
    ax2.set_title("Top 15 Bigramas - NO CLICKBAIT", fontsize=14, fontweight="bold")
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("experimento3_bigramas_frecuentes.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ============================================
    # PARTE 5: ANÁLISIS DE LONGITUD DE TITULARES
    # ============================================
    print("\n" + "=" * 70)
    print("PARTE 5: ANÁLISIS DE LONGITUD DE TITULARES")
    print("=" * 70)

    df["num_words"] = df["headline"].apply(lambda x: len(str(x).split()))
    length_stats = df.groupby("clickbait")["num_words"].describe()
    print("\n📊 Estadísticas de longitud por clase:")
    print(length_stats)

    # Visualización de longitud
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histograma
    axes[0].hist(
        df[df["clickbait"] == 0]["num_words"],
        bins=30,
        alpha=0.6,
        label="No Clickbait",
        color="steelblue",
        density=True,
    )
    axes[0].hist(
        df[df["clickbait"] == 1]["num_words"],
        bins=30,
        alpha=0.6,
        label="Clickbait",
        color="crimson",
        density=True,
    )
    axes[0].set_xlabel("Número de palabras", fontsize=12)
    axes[0].set_ylabel("Densidad", fontsize=12)
    axes[0].set_title("Distribución de Longitud de Titulares", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Boxplot
    df_box = df[["num_words", "clickbait"]].copy()
    df_box["clickbait"] = df_box["clickbait"].map({0: "No Clickbait", 1: "Clickbait"})
    sns.boxplot(data=df_box, x="clickbait", y="num_words", ax=axes[1], palette=["steelblue", "crimson"])
    axes[1].set_xlabel("Clase", fontsize=12)
    axes[1].set_ylabel("Número de palabras", fontsize=12)
    axes[1].set_title("Boxplot de Longitud por Clase", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("experimento3_longitud_titulares.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ============================================
    # RESUMEN FINAL
    # ============================================
    print("\n" + "=" * 70)
    print("RESUMEN DEL EXPERIMENTO 3")
    print("=" * 70)

    print("\n🔍 HALLAZGOS PRINCIPALES:")
    print("\n1. CARACTERÍSTICAS DEL MODELO:")
    print(f"   • Top palabra clickbait: {top_clickbait_features.iloc[0]['feature']}")
    print(f"   • Top palabra no clickbait: {top_non_clickbait_features.iloc[0]['feature']}")

    print("\n2. PATRONES LINGÜÍSTICOS:")
    # Encontrar el patrón con mayor ratio de clickbait
    max_cb_pattern = max(
        pattern_results.items(),
        key=lambda x: x[1]["clickbait"] / max(x[1]["total"], 1),
    )
    print(f"   • Patrón más asociado a clickbait: {max_cb_pattern[0]}")
    cb_ratio = max_cb_pattern[1]["clickbait"] / max(max_cb_pattern[1]["total"], 1) * 100
    print(f"     Ratio: {cb_ratio:.1f}% clickbait")

    print("\n3. PALABRAS MÁS FRECUENTES:")
    print(f"   • Clickbait: {top_clickbait_words[0][0]} ({top_clickbait_words[0][1]} ocurrencias)")
    print(f"   • No Clickbait: {top_non_clickbait_words[0][0]} ({top_non_clickbait_words[0][1]} ocurrencias)")

    print("\n4. LONGITUD DE TITULARES:")
    avg_cb = df[df["clickbait"] == 1]["num_words"].mean()
    avg_ncb = df[df["clickbait"] == 0]["num_words"].mean()
    print(f"   • Promedio Clickbait: {avg_cb:.1f} palabras")
    print(f"   • Promedio No Clickbait: {avg_ncb:.1f} palabras")

    print("\n✅ Experimento 3 completado exitosamente!")
    print("\n📁 Archivos generados:")
    print("   • experimento3_caracteristicas_importantes.png")
    print("   • experimento3_caracteristicas_importantes.csv")
    print("   • experimento3_patrones_linguisticos.png")
    print("   • experimento3_patrones_linguisticos.csv")
    print("   • experimento3_palabras_frecuentes.png")
    print("   • experimento3_bigramas_frecuentes.png")
    print("   • experimento3_longitud_titulares.png")


if __name__ == "__main__":
    main()

