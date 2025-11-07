# EXPERIMENTO 2: Clasificación con Modelos Basados en Embeddings (GloVe y BERT)
import os
import re
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm import tqdm

# Configuración de dependencias opcionales
try:
    import gensim
    from gensim.models import KeyedVectors
    from gensim.scripts.glove2word2vec import glove2word2vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("⚠️ Gensim no disponible. Instalar con: pip install gensim")

try:
    from transformers import BertTokenizer, BertModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers no disponible. Instalar con: pip install transformers torch")

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
# FUNCIONES PARA GLOVE
# ============================================

def download_glove_embeddings(glove_dir="embeddings", dim=100):
    """
    Descargar embeddings GloVe si no existen
    """
    os.makedirs(glove_dir, exist_ok=True)
    glove_file = f"glove.6B.{dim}d.txt"
    glove_path = os.path.join(glove_dir, glove_file)
    
    if os.path.exists(glove_path):
        # Verificar que el archivo no esté vacío o corrupto
        file_size = os.path.getsize(glove_path)
        if file_size > 1000:  # Al menos 1KB
            print(f"✅ GloVe embeddings ya existen: {glove_path} ({file_size / 1024 / 1024:.1f} MB)")
            return glove_path
        else:
            print(f"⚠️ Archivo GloVe existente parece estar corrupto ({file_size} bytes). Re-descargando...")
            os.remove(glove_path)
    
    print(f"📥 Descargando embeddings GloVe ({dim} dimensiones)...")
    print(f"   Esto puede tomar varios minutos dependiendo de tu conexión...")
    url = "https://nlp.stanford.edu/data/glove.6B.zip"
    zip_path = os.path.join(glove_dir, "glove.6B.zip")
    
    try:
        # Descargar con barra de progreso
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = min(int(count * block_size * 100 / total_size), 100)
                downloaded_mb = count * block_size / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(f"   Descargando... {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end="\r")
        
        urllib.request.urlretrieve(url, zip_path, progress_hook)
        print("\n✅ Descarga completa. Extrayendo...")
        
        # Verificar que el zip se descargó correctamente
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1000:
            raise Exception("El archivo descargado parece estar corrupto o vacío")
        
        # Extraer archivo específico
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            if glove_file not in zip_ref.namelist():
                raise Exception(f"Archivo {glove_file} no encontrado en el ZIP")
            zip_ref.extract(glove_file, glove_dir)
        
        # Verificar que el archivo extraído existe y tiene contenido
        if not os.path.exists(glove_path) or os.path.getsize(glove_path) < 1000:
            raise Exception("El archivo extraído parece estar corrupto o vacío")
        
        # Limpiar archivo zip
        try:
            os.remove(zip_path)
        except:
            pass  # No crítico si no se puede eliminar
        
        file_size = os.path.getsize(glove_path)
        print(f"✅ GloVe embeddings descargados: {glove_path} ({file_size / 1024 / 1024:.1f} MB)")
        return glove_path
    
    except urllib.error.URLError as e:
        print(f"❌ Error de conexión descargando GloVe: {e}")
        print("   Verifica tu conexión a internet e intenta nuevamente")
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except:
                pass
        return None
    except Exception as e:
        print(f"❌ Error descargando GloVe: {e}")
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except:
                pass
        return None


def load_glove_model(glove_path, dim=100):
    """
    Cargar modelo GloVe
    """
    if not GENSIM_AVAILABLE:
        print("❌ Gensim no disponible. No se puede cargar GloVe.")
        return None
    
    try:
        print(f"🔄 Cargando embeddings GloVe desde {glove_path}...")
        # Intentar cargar directamente (método más simple y directo)
        try:
            model = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)
            print(f"✅ GloVe cargado (directo): {len(model)} palabras, {model.vector_size} dimensiones")
            return model
        except Exception as e1:
            # Si falla, intentar con conversión a Word2Vec como fallback
            print(f"   Método directo falló: {e1}")
            print("   Intentando con conversión a Word2Vec...")
            word2vec_path = glove_path.replace(".txt", ".word2vec")
            
            if not os.path.exists(word2vec_path):
                glove2word2vec(glove_path, word2vec_path)
            
            model = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
            print(f"✅ GloVe cargado (con conversión): {len(model)} palabras, {model.vector_size} dimensiones")
            return model
    
    except Exception as e:
        print(f"❌ Error definitivo cargando GloVe: {e}")
        print("   Verifica que el archivo GloVe esté completo y no corrupto.")
        return None


def create_document_embeddings(texts, glove_model, embedding_dim=100):
    """
    Convertir lista de textos a embeddings usando GloVe
    """
    if glove_model is None:
        raise ValueError("glove_model no puede ser None")
    
    embeddings = []
    words_found = 0
    words_not_found = 0
    
    for text in tqdm(texts, desc="Creando embeddings GloVe"):
        if not isinstance(text, str):
            text = str(text)
        
        words = text.split()
        word_embeddings = []
        
        for word in words:
            if word in glove_model:
                word_embeddings.append(glove_model[word])
                words_found += 1
            else:
                words_not_found += 1
        
        if word_embeddings:
            # Promediar embeddings de palabras
            doc_embedding = np.mean(word_embeddings, axis=0)
        else:
            # Si no hay palabras conocidas, usar vector de ceros
            doc_embedding = np.zeros(embedding_dim)
        
        embeddings.append(doc_embedding)
    
    # Información estadística
    total_words = words_found + words_not_found
    if total_words > 0:
        coverage = words_found / total_words * 100
        print(f"   Cobertura de vocabulario: {coverage:.1f}% ({words_found}/{total_words} palabras encontradas)")
    
    return np.array(embeddings)


# ============================================
# FUNCIONES PARA BERT
# ============================================

def get_bert_embeddings(texts, batch_size=32, max_length=128):
    """
    Obtener embeddings BERT para una lista de textos
    """
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers no disponible. No se puede usar BERT.")
        return None
    
    try:
        print("🔄 Cargando modelo BERT...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")
        model.eval()
        
        # Detectar dispositivo
        if torch.cuda.is_available():
            model = model.cuda()
            device = "cuda"
            print(f"✅ BERT cargado en GPU (CUDA)")
        else:
            device = "cpu"
            print(f"✅ BERT cargado en CPU")
        
        print("🔄 Generando embeddings BERT...")
        
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Procesando con BERT"):
                batch_texts = texts[i : i + batch_size]
                
                # Tokenizar
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                
                if device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Obtener embeddings
                outputs = model(**inputs)
                # Usar el embedding del token [CLS] (primer token)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"❌ Error de memoria al generar embeddings BERT: {e}")
            print("   Intenta reducir el batch_size o el max_length")
        else:
            print(f"❌ Error de runtime generando embeddings BERT: {e}")
        return None
    except Exception as e:
        print(f"❌ Error generando embeddings BERT: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================
# FUNCIONES DE EVALUACIÓN
# ============================================

def evaluate_model(y_true, y_pred, model_name, X_test_emb=None):
    """
    Evaluar modelo y generar reporte completo
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"\n📊 RESULTADOS - {model_name}")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Clickbait", "Clickbait"])
    disp.plot(cmap="Blues")
    plt.title(f"Matriz de Confusión - {model_name}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


# ============================================
# FUNCIÓN PRINCIPAL
# ============================================

def main():
    print("=" * 70)
    print("EXPERIMENTO 2: MODELOS BASADOS EN EMBEDDINGS")
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
    
    # Dividir datos
    print("\n📊 Dividiendo datos...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        df["clean"], df["clickbait"], test_size=0.3, random_state=42, stratify=df["clickbait"]
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"   Train: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}")
    
    results = {}
    
    # ============================================
    # EXPERIMENTO 2.1: GLOVE
    # ============================================
    if GENSIM_AVAILABLE:
        print("\n" + "=" * 70)
        print("EXPERIMENTO 2.1: GLOVE + REGRESIÓN LOGÍSTICA")
        print("=" * 70)
        
        # Descargar y cargar GloVe
        glove_path = download_glove_embeddings(dim=100)
        if glove_path:
            glove_model = load_glove_model(glove_path, dim=100)
            
            if glove_model:
                # Crear embeddings
                print("\n🔄 Creando embeddings para los conjuntos de datos...")
                X_train_glove = create_document_embeddings(X_train, glove_model, embedding_dim=100)
                X_valid_glove = create_document_embeddings(X_valid, glove_model, embedding_dim=100)
                X_test_glove = create_document_embeddings(X_test, glove_model, embedding_dim=100)
                
                print(f"   ✅ Embeddings creados:")
                print(f"      Train: {X_train_glove.shape}")
                print(f"      Valid: {X_valid_glove.shape}")
                print(f"      Test:  {X_test_glove.shape}")
                
                # Entrenar modelo
                print("\n🤖 Entrenando modelo con GloVe...")
                param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
                lr_glove = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
                grid_glove = GridSearchCV(lr_glove, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1)
                grid_glove.fit(X_train_glove, y_train)
                
                print(f"   Mejor C: {grid_glove.best_params_['C']}")
                print(f"   Mejor F1 (CV): {grid_glove.best_score_:.4f}")
                
                # Evaluar en validación primero para verificar
                y_pred_valid = grid_glove.predict(X_valid_glove)
                print(f"\n📊 Resultados en Validación:")
                valid_f1 = f1_score(y_valid, y_pred_valid, average='weighted')
                print(f"   F1-Score (Valid): {valid_f1:.4f}")
                
                # Evaluar en test
                y_pred_glove = grid_glove.predict(X_test_glove)
                results["GloVe"] = evaluate_model(y_test, y_pred_glove, "GloVe + Regresión Logística", X_test_glove)
                results["GloVe"]["best_params"] = grid_glove.best_params_
                results["GloVe"]["valid_f1"] = valid_f1
    else:
        print("\n⚠️ Saltando experimento GloVe (Gensim no disponible)")
    
    # ============================================
    # EXPERIMENTO 2.2: BERT
    # ============================================
    if TRANSFORMERS_AVAILABLE:
        print("\n" + "=" * 70)
        print("EXPERIMENTO 2.2: BERT + REGRESIÓN LOGÍSTICA")
        print("=" * 70)
        print("⚠️ Nota: Esto puede tomar varios minutos...")
        
        # Generar embeddings BERT
        try:
            X_train_bert = get_bert_embeddings(list(X_train), batch_size=32)
            X_valid_bert = get_bert_embeddings(list(X_valid), batch_size=32)
            X_test_bert = get_bert_embeddings(list(X_test), batch_size=32)
        except Exception as e:
            print(f"❌ Error generando embeddings BERT: {e}")
            print("⚠️ Saltando experimento BERT...")
            X_train_bert = None
        
        if X_train_bert is not None and X_valid_bert is not None and X_test_bert is not None:
            print(f"\n✅ Embeddings BERT creados:")
            print(f"   Train: {X_train_bert.shape}")
            print(f"   Valid: {X_valid_bert.shape}")
            print(f"   Test:  {X_test_bert.shape}")
            
            # Entrenar modelo
            print("\n🤖 Entrenando modelo con BERT...")
            param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
            lr_bert = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
            grid_bert = GridSearchCV(lr_bert, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1)
            grid_bert.fit(X_train_bert, y_train)
            
            print(f"   Mejor C: {grid_bert.best_params_['C']}")
            print(f"   Mejor F1 (CV): {grid_bert.best_score_:.4f}")
            
            # Evaluar en validación primero para verificar
            y_pred_valid = grid_bert.predict(X_valid_bert)
            print(f"\n📊 Resultados en Validación:")
            valid_f1 = f1_score(y_valid, y_pred_valid, average='weighted')
            print(f"   F1-Score (Valid): {valid_f1:.4f}")
            
            # Evaluar en test
            y_pred_bert = grid_bert.predict(X_test_bert)
            results["BERT"] = evaluate_model(y_test, y_pred_bert, "BERT + Regresión Logística", X_test_bert)
            results["BERT"]["best_params"] = grid_bert.best_params_
            results["BERT"]["valid_f1"] = valid_f1
        else:
            print("❌ No se pudieron generar embeddings BERT correctamente")
    else:
        print("\n⚠️ Saltando experimento BERT (Transformers no disponible)")
    
    # ============================================
    # COMPARACIÓN FINAL
    # ============================================
    if results:
        print("\n" + "=" * 70)
        print("COMPARACIÓN DE MODELOS - EXPERIMENTO 2")
        print("=" * 70)
        
        # Crear tabla comparativa
        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append(
                {
                    "Modelo": model_name,
                    "Accuracy": metrics["accuracy"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1-Score": metrics["f1"],
                }
            )
        
        df_comparison = pd.DataFrame(comparison_data)
        print("\n📊 Tabla Comparativa:")
        print(df_comparison.to_string(index=False))
        
        # Guardar resultados
        df_comparison.to_csv("experimento2_resultados.csv", index=False)
        print("\n✅ Resultados guardados en: experimento2_resultados.csv")
        
        # Visualización
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(df_comparison))
        width = 0.2
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        ax.bar(x - 1.5 * width, df_comparison["Accuracy"], width, label="Accuracy", alpha=0.8, color=colors[0])
        ax.bar(x - 0.5 * width, df_comparison["Precision"], width, label="Precision", alpha=0.8, color=colors[1])
        ax.bar(x + 0.5 * width, df_comparison["Recall"], width, label="Recall", alpha=0.8, color=colors[2])
        ax.bar(x + 1.5 * width, df_comparison["F1-Score"], width, label="F1-Score", alpha=0.8, color=colors[3])
        
        ax.set_xlabel("Modelos", fontsize=12)
        ax.set_ylabel("Puntuación", fontsize=12)
        ax.set_title("Comparación de Modelos Basados en Embeddings", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(df_comparison["Modelo"])
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        plt.savefig("experimento2_comparacion.png", dpi=300, bbox_inches="tight")
        plt.show()
        
        print("\n✅ Experimento 2 completado exitosamente!")
    else:
        print("\n❌ No se pudieron ejecutar experimentos.")
        print("   Instala las dependencias necesarias:")
        print("   pip install gensim transformers torch")


if __name__ == "__main__":
    main()

