"""
Experimento 2: Clasificación con Modelos Basados en Embeddings
Implementa GloVe y BERT para comparar con modelos clásicos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

try:
    import gensim
    from gensim.models import Word2Vec
    from gensim.scripts.glove2word2vec import glove2word2vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("Gensim no disponible. Instalar con: pip install gensim")

try:
    from transformers import BertTokenizer, BertModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers no disponible. Instalar con: pip install transformers torch")

class EmbeddingsExperiment:
    """
    Clase para experimentos con embeddings avanzados
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def load_glove_embeddings(self, glove_path=None):
        """
        Cargar embeddings GloVe preentrenados
        """
        if not GENSIM_AVAILABLE:
            print("Gensim no disponible. Saltando GloVe.")
            return None
            
        if glove_path is None:
            print("Ruta de GloVe no especificada. Usando Word2Vec como alternativa.")
            return None
            
        try:
            # Convertir GloVe a formato Word2Vec
            word2vec_path = glove_path.replace('.txt', '.word2vec')
            glove2word2vec(glove_path, word2vec_path)
            
            # Cargar modelo
            model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
            print(f"GloVe cargado: {len(model)} palabras, {model.vector_size} dimensiones")
            return model
        except Exception as e:
            print(f"Error cargando GloVe: {e}")
            return None
    
    def create_word2vec_embeddings(self, texts, vector_size=100, window=5, min_count=1):
        """
        Crear embeddings Word2Vec desde cero
        """
        if not GENSIM_AVAILABLE:
            print("Gensim no disponible. Saltando Word2Vec.")
            return None
            
        print("Entrenando Word2Vec...")
        
        # Tokenizar textos
        tokenized_texts = [text.split() for text in texts]
        
        # Entrenar Word2Vec
        model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            seed=self.random_state
        )
        
        print(f"Word2Vec entrenado: {len(model.wv)} palabras, {model.vector_size} dimensiones")
        return model
    
    def text_to_embedding(self, text, embedding_model, method='average'):
        """
        Convertir texto a vector de embedding
        """
        if embedding_model is None:
            return None
            
        words = text.split()
        vectors = []
        
        for word in words:
            if word in embedding_model:
                vectors.append(embedding_model[word])
        
        if not vectors:
            # Si no hay palabras conocidas, devolver vector de ceros
            return np.zeros(embedding_model.vector_size)
        
        if method == 'average':
            return np.mean(vectors, axis=0)
        elif method == 'sum':
            return np.sum(vectors, axis=0)
        else:
            return np.mean(vectors, axis=0)
    
    def experiment_word2vec(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Experimento con Word2Vec
        """
        print("\n" + "="*50)
        print("EXPERIMENTO: Word2Vec + Regresión Logística")
        print("="*50)
        
        if not GENSIM_AVAILABLE:
            print("Gensim no disponible. Saltando experimento Word2Vec.")
            return {}
        
        # Crear embeddings Word2Vec
        all_texts = list(X_train) + list(X_val) + list(X_test)
        w2v_model = self.create_word2vec_embeddings(all_texts)
        
        if w2v_model is None:
            return {}
        
        # Convertir textos a vectores
        print("Convirtiendo textos a vectores...")
        X_train_vec = np.array([self.text_to_embedding(text, w2v_model) for text in X_train])
        X_val_vec = np.array([self.text_to_embedding(text, w2v_model) for text in X_val])
        X_test_vec = np.array([self.text_to_embedding(text, w2v_model) for text in X_test])
        
        # Entrenar clasificador
        print("Entrenando clasificador...")
        clf = LogisticRegression(random_state=self.random_state, max_iter=1000)
        clf.fit(X_train_vec, y_train)
        
        # Evaluar
        y_pred = clf.predict(X_test_vec)
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"Word2Vec + Regresión Logística:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  F1-Score: {results['f1']:.4f}")
        
        self.models['word2vec'] = {'model': w2v_model, 'classifier': clf}
        return results
    
    def experiment_bert(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Experimento con BERT (versión simplificada)
        """
        print("\n" + "="*50)
        print("EXPERIMENTO: BERT + Regresión Logística")
        print("="*50)
        
        if not TRANSFORMERS_AVAILABLE:
            print("Transformers no disponible. Saltando experimento BERT.")
            return {}
        
        try:
            # Cargar tokenizador y modelo BERT
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')
            model.eval()
            
            def get_bert_embeddings(texts, batch_size=32):
                """Obtener embeddings BERT para una lista de textos"""
                embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    
                    # Tokenizar
                    inputs = tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors='pt'
                    )
                    
                    # Obtener embeddings
                    with torch.no_grad():
                        outputs = model(**inputs)
                        # Usar el embedding del token [CLS] (primer token)
                        batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                        embeddings.extend(batch_embeddings)
                
                return np.array(embeddings)
            
            print("Obteniendo embeddings BERT...")
            X_train_bert = get_bert_embeddings(list(X_train))
            X_val_bert = get_bert_embeddings(list(X_val))
            X_test_bert = get_bert_embeddings(list(X_test))
            
            # Entrenar clasificador
            print("Entrenando clasificador...")
            clf = LogisticRegression(random_state=self.random_state, max_iter=1000)
            clf.fit(X_train_bert, y_train)
            
            # Evaluar
            y_pred = clf.predict(X_test_bert)
            
            results = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            print(f"BERT + Regresión Logística:")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  F1-Score: {results['f1']:.4f}")
            
            self.models['bert'] = {'tokenizer': tokenizer, 'model': model, 'classifier': clf}
            return results
            
        except Exception as e:
            print(f"Error en experimento BERT: {e}")
            return {}
    
    def compare_embeddings(self, results_dict):
        """
        Comparar resultados de diferentes métodos de embedding
        """
        if not results_dict:
            print("No hay resultados para comparar")
            return
        
        print("\n" + "="*60)
        print("COMPARACIÓN DE MÉTODOS DE EMBEDDING")
        print("="*60)
        
        # Crear tabla de comparación
        comparison_data = []
        for method, results in results_dict.items():
            comparison_data.append({
                'Método': method,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print("\nComparación de Métodos:")
        print(df_comparison.round(4))
        
        # Gráfico de comparación
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(df_comparison))
        width = 0.2
        
        ax.bar(x - width, df_comparison['Accuracy'], width, label='Accuracy', alpha=0.8)
        ax.bar(x, df_comparison['Precision'], width, label='Precision', alpha=0.8)
        ax.bar(x + width, df_comparison['Recall'], width, label='Recall', alpha=0.8)
        ax.bar(x + 2*width, df_comparison['F1-Score'], width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Métodos de Embedding')
        ax.set_ylabel('Puntuación')
        ax.set_title('Comparación de Métodos de Embedding')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(df_comparison['Método'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('embeddings_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df_comparison

def main():
    """
    Función principal para ejecutar experimentos de embeddings
    """
    print("EXPERIMENTO 2: MODELOS BASADOS EN EMBEDDINGS")
    print("="*60)
    
    # Cargar datos de ejemplo
    try:
        df = pd.read_csv('./data/clickbait_sample.csv')
        print(f"Dataset cargado: {df.shape[0]} titulares")
    except FileNotFoundError:
        print("Error: No se encontró el dataset. Ejecuta main.py primero.")
        return
    
    # Preprocesamiento básico
    from clickbait_detection import ClickbaitDetector
    detector = ClickbaitDetector()
    df['processed_text'] = df['headline'].apply(detector.preprocess_text)
    
    # División de datos
    X_train, X_val, X_test, y_train, y_val, y_test = detector.split_data(df)
    
    # Inicializar experimento de embeddings
    embeddings_exp = EmbeddingsExperiment()
    
    # Ejecutar experimentos
    results = {}
    
    # Word2Vec
    w2v_results = embeddings_exp.experiment_word2vec(X_train, X_val, X_test, y_train, y_val, y_test)
    if w2v_results:
        results['Word2Vec'] = w2v_results
    
    # BERT
    bert_results = embeddings_exp.experiment_bert(X_train, X_val, X_test, y_train, y_val, y_test)
    if bert_results:
        results['BERT'] = bert_results
    
    # Comparar resultados
    if results:
        embeddings_exp.compare_embeddings(results)
        print("\n✓ Experimentos de embeddings completados!")
    else:
        print("\n✗ No se pudieron ejecutar experimentos de embeddings.")
        print("Instala las dependencias necesarias:")
        print("pip install gensim transformers torch")

if __name__ == "__main__":
    main()
