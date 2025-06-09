# Proyecto_RI
Proyecto_RI_2025A
# Descripción
Este proyecto implementa un sistema básico de recuperación de información que permite:
 - Indexar un corpus de documentos en texto plano.
 - Realizar consultas de texto libre.
 - Usar modelos vectoriales TF-IDF y BM25 para ranking.
 - Evaluar la calidad de los resultados con métricas: Precisión, Recall y MAP.
# Descripción del Corpus
El corpus utilizado para la ejecución del sistema es **"beir/cqadupstack/webmasters"**, parte del grupo **BEIR** (Benchmarking IR), una colección de 18 datasets diseñados para evaluar modelos de recuperación de información en distintos dominios.
En este caso específico, el corpus corresponde a **CQADupStack**, un dataset que busca identificar preguntas duplicadas en foros. Se empleó el sub-dataset **"webmasters"**, correspondiente al foro *Webmasters*, que contiene aproximadamente **17,000 documentos** y **506 consultas**, todas basadas en preguntas reales del foro.
# Requisitos
 - Tener instalado Python.
 - Librerías Python necesarias:
  pip install numpy pandas nltk scikit-learn
Cómo ejecutar el sistema
# Preparar entorno 
 Asegúrate de estar ubicado en la carpeta correcta donde están los archivos .py y los archivos de entrada (.tsv, .jsonl).
 La ejecución siempre debe hacerse con el comando:
 - python nombre_del_archivo.py
# Archivos principales
 - SistemaRI.py Archivo principal.
 - test.tsv Relevancia de documentos.
 - corpus_preprocesado_nltk.jsonl  Corpus ya procesado
 - corpus.jsonl Corpus original.
# Resultados
 - Se imprimen las métricas Precision, Recall, Average Precision para cada consulta.
 - Se muestra el Mean Average Precision (MAP) general.
 - Se visualiza un ranking de documentos recuperados por consulta con su score.
