import json
import nltk
import pandas as pd
import math
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Descargar recursos necesarios de NLTK para el procesamiento de texto
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Configuración inicial de variables globales
stop_words = set(stopwords.words('english'))  # Conjunto de stopwords en inglés
lemmatizer = WordNetLemmatizer()  # Lematizador para normalizar palabras
indice_invertido = defaultdict(dict)  # Estructura para el índice invertido
doc_lengths = {}  # Almacena la longitud de cada documento
documents = {}  # Almacena los documentos procesados
original_documents = {}  # Almacena los documentos originales
total_length = 0  # Longitud total de todos los documentos combinados
queries = {}  # Diccionario para almacenar las consultas de evaluación

def load_documents():
    """
    Carga los documentos desde un archivo JSONL.
    Cada línea del archivo representa un documento con campos '_id', 'text' y 'title'.
    """
    file_path = r'C:\Users\Andres\Downloads\cqadupstack\cqadupstack\webmasters\corpus.jsonl'
    corpus = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line.strip())
                    # Asegurar que cada documento tenga los campos mínimos requeridos
                    if '_id' not in doc:
                        doc['_id'] = str(len(corpus) + 1)
                    if 'text' not in doc:
                        doc['text'] = ''
                    if 'title' not in doc:
                        doc['title'] = ''
                    corpus.append(doc)
                except json.JSONDecodeError as e:
                    print(f"Error al decodificar línea JSON: {e}")
                    continue
        
        print(f"Se cargaron {len(corpus)} documentos correctamente")
        return corpus
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {file_path}")
        return []
    except Exception as e:
        print(f"Error al leer el archivo: {str(e)}")
        return []

def load_queries():
    """
    Carga las consultas de evaluación desde un archivo JSONL.
    Las consultas se almacenan en el diccionario global 'queries'.
    """
    queries_path = r'C:\Users\Andres\Downloads\cqadupstack\cqadupstack\webmasters\queries.jsonl'
    
    try:
        with open(queries_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    query = json.loads(line.strip())
                    if '_id' in query and 'text' in query:
                        queries[str(query['_id'])] = query['text']
                except json.JSONDecodeError as e:
                    print(f"Error al decodificar línea JSON de consulta: {e}")
                    continue
        
        print(f"Se cargaron {len(queries)} consultas correctamente")
        return True
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de consultas en {queries_path}")
        return False
    except Exception as e:
        print(f"Error al leer el archivo de consultas: {str(e)}")
        return False

def load_qrels():
    """
    Carga los QRELS (relevancias de consultas) desde un archivo TSV.
    Los QRELS indican qué documentos son relevantes para cada consulta.
    """
    qrels_path = r'C:\Users\Andres\Downloads\cqadupstack\cqadupstack\webmasters\qrels\test.tsv'
    
    try:
        df = pd.read_csv(
            qrels_path,
            sep='\t',
            header=0,
            names=['query-id', 'corpus-id', 'score'],
            engine='python'
        )
        
        print(f"Se cargaron {len(df)} QRELS correctamente (archivo TSV)")
        return df
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo QRELS en {qrels_path}")
        return None
    except Exception as e:
        print(f"Error al leer el archivo QRELS: {str(e)}")
        return None

def preprocess(text):
    """
    Preprocesa el texto: tokeniza, elimina stopwords y lematiza.
    Devuelve una lista de tokens procesados.
    """
    tokens = regexp_tokenize(text.lower(), pattern=r'\w[a-z]+')
    filtered = [w for w in tokens if w not in stop_words]
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered]
    return lemmatized

def preprocess_for_tfidf(text):
    """
    Preprocesa el texto para TF-IDF: similar a preprocess pero devuelve un string.
    """
    tokens = regexp_tokenize(text.lower(), pattern=r'\w[a-z]+')
    filtered = [w for w in tokens if w not in stop_words]
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered]
    return " ".join(lemmatized)

def procesar_corpus(corpus):
    """
    Procesa todo el corpus: aplica preprocesamiento a cada documento
    y guarda los resultados en estructuras globales y archivos.
    """
    global total_length, doc_lengths, documents, original_documents
    
    preprocessed_corpus = []
    for doc in corpus:
        # Procesar el texto y título de cada documento
        processed_text = preprocess(doc["text"])
        processed_title = preprocess(doc["title"])
        
        new_doc = {
            "doc_id": doc["_id"],
            "text_tokens": processed_text,
            "title_tokens": processed_title,
        }
        preprocessed_corpus.append(new_doc)
        
        # Almacenar en estructuras globales
        doc_id = str(doc["_id"])
        documents[doc_id] = processed_text + processed_title
        doc_lengths[doc_id] = len(documents[doc_id])
        total_length += doc_lengths[doc_id]
        original_documents[doc_id] = {
            "text": doc.get("text", ""),
            "title": doc.get("title", "")
        }
    
    # Guardar corpus preprocesado en archivo
    with open("corpus_preprocesado_nltk.jsonl", "w", encoding="utf-8") as f:
        for doc in preprocessed_corpus:
            json.dump(doc, f)
            f.write("\n")
    
    return preprocessed_corpus

def construir_indice_invertido():
    """
    Construye un índice invertido a partir del corpus preprocesado.
    El índice mapea términos a los documentos que los contienen y su frecuencia.
    """
    global indice_invertido
    
    with open("corpus_preprocesado_nltk.jsonl", 'r', encoding='utf-8') as f:
        for linea in f:
            doc = json.loads(linea)
            doc_id = doc['doc_id']
            tokens = doc['text_tokens'] + doc['title_tokens']
            
            # Contar frecuencia de términos por documento
            frecuencia = Counter(tokens)
            for palabra, freq in frecuencia.items():
                indice_invertido[palabra][doc_id] = freq
    
    # Guardar índice invertido en archivo
    with open("indice_invertido.json", "w", encoding="utf-8") as f:
        json.dump(indice_invertido, f)

def buscar_tfidf(consulta):
    """
    Realiza una búsqueda usando el modelo TF-IDF.
    Calcula la similitud coseno entre la consulta y los documentos.
    """
    corpus = []
    with open("corpus_preprocesado_nltk.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            corpus.append(json.loads(line))
    
    # Preparar documentos para TF-IDF
    docs_as_text = [" ".join(doc["text_tokens"] + doc["title_tokens"]) for doc in corpus]
    doc_ids = [doc["doc_id"] for doc in corpus]
    
    # Crear matriz TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs_as_text)
    
    # Procesar consulta y calcular similitudes
    consulta_preprocesada = preprocess_for_tfidf(consulta)
    consulta_vectorizada = vectorizer.transform([consulta_preprocesada])
    similitudes = cosine_similarity(consulta_vectorizada, tfidf_matrix).flatten()
    
    # Ordenar resultados por similitud
    resultados_ordenados = sorted(
        zip(doc_ids, similitudes),
        key=lambda x: x[1],
        reverse=True
    )
    
    print("\nResultados de búsqueda (TF-IDF):")
    print(f"Consulta: '{consulta}'\n")
    
    # Mostrar resultados
    if resultados_ordenados and resultados_ordenados[0][1] > 0:
        doc_id, score = resultados_ordenados[0]
        original = original_documents.get(str(doc_id), {})
        print("Documento más relevante:")
        print(f"Título: {original.get('title', '[sin título]')}")
        print(f"Texto: {original.get('text', '[sin contenido]')}")
        print(f"Similitud: {score:.4f}\n")
    
    print("\nTop 5 documentos relevantes:")
    found_relevant = False
    for doc_id, score in resultados_ordenados[:5]:
        if score > 0:
            found_relevant = True
            original = original_documents.get(str(doc_id), {})
            print(f"\nDocumento ID: {doc_id}")
            print(f"Título: {original.get('title', '[sin título]')}")
            print(f"Similitud: {score:.4f}")
    
    if not found_relevant:
        print("No hay documentos relevantes para la consulta.")

def buscar_bm25(consulta):
    """
    Realiza una búsqueda usando el modelo BM25.
    Calcula scores BM25 para cada documento relevante a la consulta.
    """
    avg_dl = total_length / len(doc_lengths) if doc_lengths else 0
    query_tokens = preprocess(consulta)
    
    N = len(documents)
    scores = defaultdict(float)

    # Calcular score BM25 para cada término de la consulta
    for term in query_tokens:
        if term in indice_invertido:
            df = len(indice_invertido[term])  # Document frequency
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)  # IDF suavizado

            for doc_id, freq in indice_invertido[term].items():
                tf = freq  # Term frequency
                dl = doc_lengths[doc_id]  # Document length
                # Fórmula BM25 con parámetros k1=1.5 y b=0.75
                score = idf * tf * (1.5 + 1) / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avg_dl))
                scores[doc_id] += score
    
    # Ordenar resultados por score
    resultados = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\nResultados de búsqueda (BM25):")
    print(f"Consulta: '{consulta}'\n")
    
    # Mostrar resultados
    if resultados and resultados[0][1] > 0:
        doc_id, score = resultados[0]
        original = original_documents.get(doc_id, {})
        print("Documento más relevante:")
        print(f"Título: {original.get('title', '[sin título]')}")
        print(f"Texto: {original.get('text', '[sin contenido]')}")
        print(f"Score BM25: {score:.4f}\n")
    
    print("\nTop 5 documentos relevantes:")
    found_relevant = False
    for doc_id, score in resultados[:5]:
        if score > 0:
            found_relevant = True
            original = original_documents.get(doc_id, {})
            print(f"\nDocumento ID: {doc_id}")
            print(f"Título: {original.get('title', '[sin título]')}")
            print(f"Score BM25: {score:.4f}")
    
    if not found_relevant:
        print("No hay documentos relevantes para la consulta.")

def evaluar_sistema(qrels_df):
    """
    Evalúa el sistema usando los QRELS.
    Calcula métricas de evaluación: Precision@10, Recall@10 y MAP.
    """
    if qrels_df is None:
        print("No se pudo realizar la evaluación por falta de QRELS")
        return
    
    if not queries:
        print("No se pudieron cargar las consultas. No se puede evaluar.")
        return
    
    # Procesar QRELS para obtener documentos relevantes por consulta
    relevantes_por_query = defaultdict(dict)
    
    try:
        for _, row in qrels_df.iterrows():
            query_id = str(row['query-id'])
            doc_id = str(row['corpus-id'])
            score = float(row['score'])
            
            if score > 0:
                if query_id in queries:
                    if query_id not in relevantes_por_query:
                        relevantes_por_query[query_id] = {
                            'text': queries[query_id],
                            'relevant_docs': set()
                        }
                    relevantes_por_query[query_id]['relevant_docs'].add(doc_id)
    except Exception as e:
        print(f"Error al procesar QRELS: {e}")
        return
    
    # Cargar corpus preprocesado
    corpus = []
    with open("corpus_preprocesado_nltk.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            corpus.append(json.loads(line))
    
    # Preparar documentos para TF-IDF (usado para evaluación)
    docs_as_text = [" ".join(doc["text_tokens"] + doc["title_tokens"]) for doc in corpus]
    doc_ids = [str(doc["doc_id"]) for doc in corpus]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs_as_text)
    
    def evaluar_query(query_text, relevantes_ids, top_k=10):
        """
        Evalúa una consulta individual calculando métricas de rendimiento.
        """
        consulta_preprocesada = preprocess_for_tfidf(query_text)
        consulta_vector = vectorizer.transform([consulta_preprocesada])
        similitudes = cosine_similarity(consulta_vector, tfidf_matrix).flatten()

        resultados_ordenados = sorted(
            zip(doc_ids, similitudes),
            key=lambda x: x[1],
            reverse=True
        )

        top_resultados = resultados_ordenados[:top_k]
        pred_ids = [doc_id for doc_id, _ in top_resultados]

        # Calcular métricas
        verdaderos_positivos = len(set(pred_ids) & relevantes_ids)
        precision = verdaderos_positivos / len(pred_ids) if pred_ids else 0
        recall = verdaderos_positivos / len(relevantes_ids) if relevantes_ids else 0
        
        # Calcular Average Precision (AP)
        num_relevantes_encontrados = 0
        suma_precision = 0.0
        for i, (doc_id, _) in enumerate(top_resultados):
            if doc_id in relevantes_ids:
                num_relevantes_encontrados += 1
                precision_at_i = num_relevantes_encontrados / (i + 1)
                suma_precision += precision_at_i
        
        ap = suma_precision / len(relevantes_ids) if relevantes_ids else 0

        return {
            'precision': precision,
            'recall': recall,
            'ap': ap,
            'pred_ids': pred_ids,
            'similitudes': [score for _, score in top_resultados]
        }
    
    print("\nEvaluación del sistema:\n")
    map_total = 0.0
    num_consultas_evaluadas = 0
    
    # Evaluar cada consulta (hasta 7 para no saturar la salida)
    for query_id, data in list(relevantes_por_query.items())[:7]:
        query_text = data['text']
        relevantes_ids = data['relevant_docs']
        
        if not relevantes_ids:
            continue
            
        resultados = evaluar_query(query_text, relevantes_ids)
        
        # Mostrar resultados por consulta
        print(f"Consulta ID: {query_id}")
        print(f"Texto: '{query_text}'")
        print(f"Documentos relevantes esperados: {len(relevantes_ids)}")
        print(f"Precision@10: {resultados['precision']:.5f}")
        print(f"Recall@10: {resultados['recall']:.4f}")
        print(f"Average Precision: {resultados['ap']:.5f}")
        
        print("\nTop 5 documentos recuperados:")
        for i, (doc_id, score) in enumerate(zip(resultados['pred_ids'], resultados['similitudes'])):
            if i >= 5:
                break
            relevante = "Relevante" if doc_id in relevantes_ids else "No relevante"
            original = original_documents.get(doc_id, {})
            print(f"{i+1}. {relevante} DocID: {doc_id} - Score: {score:.4f}")
            print(f"   Título: {original.get('title', '[sin título]')}")
        
        print("\n--------------------------------------------------\n")
        
        map_total += resultados['ap']
        num_consultas_evaluadas += 1
    
    # Calcular y mostrar MAP (Mean Average Precision)
    if num_consultas_evaluadas > 0:
        map_final = map_total / num_consultas_evaluadas
        print(f"\nMean Average Precision (MAP) del sistema: {map_final:.6f}")
    else:
        print("No se pudo calcular MAP - No hay consultas con documentos relevantes")

def ejecutar_flujo_completo():
    """
    Función principal que ejecuta todo el flujo del sistema:
    1. Carga de datos
    2. Procesamiento
    3. Búsqueda interactiva
    4. Evaluación automática
    """
    print("Iniciando sistema de recuperación de información...\n")
    
    # 1. Cargar documentos
    print("Cargando documentos desde archivo...")
    corpus = load_documents()
    
    if not corpus:
        print("No se pudo cargar el corpus. Saliendo del programa.")
        return
    
    # 2. Cargar consultas (para evaluación)
    print("Cargando consultas desde archivo...")
    load_queries()
    
    # 3. Procesar corpus
    print("Procesando corpus...")
    procesar_corpus(corpus)
    construir_indice_invertido()
    print("Corpus cargado y procesado exitosamente.\n")
    
    # 4. Cargar QRELS (para evaluación)
    print("Cargando QRELS desde archivo...")
    qrels_df = load_qrels()
    
    # 5. Búsqueda interactiva
    print("\nModo de búsqueda interactiva (ingrese 'salir' para terminar)")
    while True:
        consulta_usuario = input("\nIngrese su consulta de búsqueda (ingrese 'salir' para terminar): ").strip()
        
        if consulta_usuario.lower() == 'salir':
            break
            
        if not consulta_usuario:
            print("Por favor ingrese una consulta válida.")
            continue
            
        # Ejecutar ambas búsquedas (TF-IDF y BM25)
        print(f"\nResultados para: '{consulta_usuario}'")
        
        # Búsqueda TF-IDF
        buscar_tfidf(consulta_usuario)
        
        # Búsqueda BM25
        buscar_bm25(consulta_usuario)
    
    # 6. Evaluación automática del sistema
    print("\nEvaluando automáticamente el sistema con QRELS...")
    evaluar_sistema(qrels_df)
    
    print("\nProceso completado.")

if __name__ == "__main__":
    ejecutar_flujo_completo()