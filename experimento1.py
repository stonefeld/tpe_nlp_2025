# EXPERIMENTO 1: Clasificación con modelo clásico (linea base)
import re

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm import tqdm

nltk.download("stopwords", raise_on_error=True)
nltk.download("wordnet", raise_on_error=True)
nltk.download("omw-1.4", raise_on_error=True)
nltk.download("averaged_perceptron_tagger_eng", raise_on_error=True)

# Primero cargamos el dataset
df = pd.read_csv("assets/clickbait_data.csv")  # ejemplo: headline, clickbait
df = df.dropna(subset=["headline", "clickbait"])

print("Shape:", df.shape)
print(df.head())

# Preprocesamos el texto
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
        return wordnet.NOUN  # por defecto


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    lemmas = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged]
    return " ".join(lemmas)


tqdm.pandas(desc="Preprocesando titulares")
df["clean"] = df["headline"].progress_apply(clean_text)

# Dividimos los datos en train, valid y test
# Hacemos 70% train, 15% valid, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(df["clean"], df["clickbait"], test_size=0.3, random_state=42, stratify=df["clickbait"])
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Train: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}")

# Vectorizamos usando BoW y TF-IDF con unigramas y bigramas
vectorizers = {
    "BoW": CountVectorizer(max_features=1000, ngram_range=(1, 2)),
    "TF-IDF": TfidfVectorizer(max_features=1000, ngram_range=(1, 2)),
}

results = {}

for name, vectorizer in vectorizers.items():
    print(f"\n=== Entrenando modelo con {name} ===")

    # Ajuste de vectorizador
    X_train_vec = vectorizer.fit_transform(X_train)
    X_valid_vec = vectorizer.transform(X_valid)
    X_test_vec = vectorizer.transform(X_test)

    # Entrenamos un modelo de regresión logística con búsqueda del hiperparámetro C
    param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
    lr = LogisticRegression(max_iter=1000, solver="liblinear")
    grid = GridSearchCV(lr, param_grid, cv=3, scoring="f1", n_jobs=-1)
    grid.fit(X_train_vec, y_train)

    best_lr = grid.best_estimator_
    print(f"Mejor C: {grid.best_params_['C']}")

    # Evaluamos en el conjunto de validación
    y_pred = best_lr.predict(X_valid_vec)
    report = classification_report(y_valid, y_pred, output_dict=True)
    results[name] = report

    print(classification_report(y_valid, y_pred, digits=3))

    # Matriz de confusión
    cm = confusion_matrix(y_valid, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Clickbait", "Clickbait"])
    disp.plot(cmap="Blues")
    plt.title(f"Matriz de Confusión ({name})")
    plt.show()

# Comparación de resultados
metrics = ["precision", "recall", "f1-score", "accuracy"]
df_results = pd.DataFrame(
    {
        name: {
            "Precision": results[name]["weighted avg"]["precision"],
            "Recall": results[name]["weighted avg"]["recall"],
            "F1-Score": results[name]["weighted avg"]["f1-score"],
            "Accuracy": results[name]["accuracy"],
        }
        for name in results
    }
).T

print("\n=== Resultados Comparativos ===")
print(df_results)

sns.heatmap(df_results, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("Comparación BoW vs TF-IDF - Regresión Logística")
plt.tight_layout()
plt.savefig("experimento1_comparacion.png", dpi=300, bbox_inches="tight")
plt.show()

# Guardar resultados en CSV
df_results.index.name = "Modelo"
df_results.reset_index(inplace=True)
df_results.to_csv("experimento1_resultados.csv", index=False)
print("\n✅ Resultados guardados en: experimento1_resultados.csv")
