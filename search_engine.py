# Importación de bibliotecas necesarias
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

# Descarga de recursos necesarios de NLTK (tokenizador y lista de stopwords)
nltk.download('punkt')
nltk.download('stopwords')

# Función para preprocesar texto: minúsculas, quitar puntuación, eliminar stopwords, aplicar stemming
def preprocess_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = text.translate(str.maketrans('', '', string.punctuation))  # Eliminar puntuación
    tokens = word_tokenize(text)  # Tokenizar el texto
    tokens = [w for w in tokens if w not in stopwords.words('english')]  # Eliminar stopwords
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(w) for w in tokens]  # Aplicar stemming
    return tokens

# Función para cargar los documentos desde un archivo de texto
def load_documents():
    file_path = r'C:\Users\ASUS\Downloads\01_corpus_turismo_500_en.txt'  # Ruta al archivo de entrada

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()  # Leer todas las líneas del archivo
    documents = [line.strip() for line in lines if line.strip()]  # Eliminar líneas vacías
    filenames = [f"Línea {i+1}" for i in range(len(documents))]  # Etiquetas de líneas
    return documents, filenames

# Representación de documentos con TF-IDF
def tfidf_representation(docs):
    vectorizer = TfidfVectorizer(stop_words='english')  # Eliminar stopwords automáticamente
    tfidf_matrix = vectorizer.fit_transform(docs)  # Transformar documentos a vectores TF-IDF
    return tfidf_matrix, vectorizer

# Entrenar un modelo Word2Vec usando los documentos tokenizados
def word2vec_representation(docs):
    tokenized = [preprocess_text(doc) for doc in docs]  # Preprocesar cada documento
    model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4)  # Entrenar modelo
    return model, tokenized

# Calcular el vector promedio de las palabras de un documento
def average_word_vector(words, model, dim):
    vec = np.zeros(dim)  # Inicializar vector de ceros
    count = 0
    for word in words:
        if word in model.wv:  # Solo usar palabras presentes en el vocabulario del modelo
            vec += model.wv[word]
            count += 1
    return vec / count if count > 0 else vec  # Promedio o vector cero si no hay palabras válidas

# Buscar documentos similares usando representación TF-IDF
def search_tfidf(query, tfidf_matrix, vectorizer, filenames, documents):
    query_vec = vectorizer.transform([query])  # Vectorizar la consulta
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()  # Calcular similitud coseno
    ranked = sim_scores.argsort()[::-1]  # Ordenar por similitud descendente
    return [(i, filenames[i], documents[i], sim_scores[i]) for i in ranked[:5]]  # Retornar top 5 resultados

# Buscar documentos similares usando representación Word2Vec
def search_word2vec(query, model, tokenized_docs, filenames, documents):
    query_tokens = preprocess_text(query)  # Preprocesar la consulta
    query_vec = average_word_vector(query_tokens, model, model.vector_size)  # Obtener vector promedio
    similarities = []
    for tokens in tokenized_docs:
        doc_vec = average_word_vector(tokens, model, model.vector_size)  # Vector promedio del documento
        sim = cosine_similarity([query_vec], [doc_vec])[0][0]  # Calcular similitud coseno
        similarities.append(sim)
    ranked = np.argsort(similarities)[::-1]  # Ordenar por similitud descendente
    return [(i, filenames[i], documents[i], similarities[i]) for i in ranked[:5]]  # Retornar top 5

# Función principal del programa
def main():
    query = input("Consulta de búsqueda: ").strip()  # Solicitar consulta al usuario

    documents, filenames = load_documents()  # Cargar documentos

    print("\nIndexando con TF-IDF...")
    tfidf_matrix, tfidf_vectorizer = tfidf_representation(documents)  # Generar matriz TF-IDF

    print("Entrenando Word2Vec...")
    w2v_model, tokenized_docs = word2vec_representation(documents)  # Entrenar Word2Vec

    print("\nResultados con TF-IDF:")
    for idx, fname, doc, score in search_tfidf(query, tfidf_matrix, tfidf_vectorizer, filenames, documents):
        print(f"{fname} (Doc #{idx+1}) - Score: {score:.4f}")
        print(f"→ {doc}\n")

    print("\nResultados con Word2Vec:")
    for idx, fname, doc, score in search_word2vec(query, w2v_model, tokenized_docs, filenames, documents):
        print(f"{fname} (Doc #{idx+1}) - Score: {score:.4f}")
        print(f"→ {doc}\n")

# Ejecutar la función principal si el script es ejecutado directamente
if __name__ == "__main__":
    main()
