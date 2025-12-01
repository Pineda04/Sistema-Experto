"""
Sistema RAG para consultar documentos oficiales de la UNAH
Utiliza LLaMA 3.1 a través de Ollama
"""

import os
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class RAGSystemUNAH:
    def __init__(self, documentos_path: str = "./documentos", modelo: str = "llama3.1"):
        """
        Inicializa el sistema RAG
        
        Args:
            documentos_path: Ruta a la carpeta con documentos de la UNAH
            modelo: Nombre del modelo en Ollama (llama3.1, mistral, gemma:2b)
        """
        self.documentos_path = documentos_path
        self.modelo_name = modelo
        self.vectorstore = None
        self.qa_chain = None
        
        # Configurar embeddings (modelo para convertir texto a vectores)
        print("Cargando modelo de embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # Configurar LLM con Ollama
        print(f"Conectando con Ollama - Modelo: {self.modelo_name}...")
        self.llm = Ollama(
            model=self.modelo_name,
            temperature=0.3  # Baja temperatura para respuestas más precisas
        )
    
    def cargar_documentos(self) -> List:
        """
        Carga todos los documentos PDF y TXT de la carpeta especificada
        """
        print(f"Cargando documentos desde {self.documentos_path}...")
        
        documentos = []
        
        # Cargar PDFs
        try:
            pdf_loader = DirectoryLoader(
                self.documentos_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            documentos.extend(pdf_loader.load())
        except Exception as e:
            print(f"Error cargando PDFs: {e}")
        
        # Cargar archivos de texto
        try:
            txt_loader = DirectoryLoader(
                self.documentos_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                show_progress=True
            )
            documentos.extend(txt_loader.load())
        except Exception as e:
            print(f"Error cargando TXTs: {e}")
        
        print(f"Total de documentos cargados: {len(documentos)}")
        return documentos
    
    def dividir_documentos(self, documentos: List) -> List:
        """
        Divide los documentos en chunks más pequeños para mejor recuperación
        """
        print("Dividiendo documentos en fragmentos...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Tamaño de cada fragmento
            chunk_overlap=200,  # Solapamiento entre fragmentos
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documentos)
        print(f"Total de fragmentos creados: {len(chunks)}")
        return chunks
    
    def crear_vectorstore(self, chunks: List):
        """
        Crea la base de datos vectorial con ChromaDB
        """
        print("Creando base de datos vectorial...")
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        print("Base de datos vectorial creada exitosamente!")
    
    def crear_prompt_template(self) -> PromptTemplate:
        """
        Crea el prompt maestro para el modelo
        """
        template = """Eres un experto en normativas y reglamentos de la Universidad Nacional Autónoma de Honduras (UNAH).
Tu tarea es analizar consultas y proporcionar respuestas precisas basadas ÚNICAMENTE en los documentos oficiales de la UNAH.

CONTEXTO DE LOS DOCUMENTOS:
{context}

PREGUNTA DEL USUARIO:
{question}

INSTRUCCIONES PARA TU RESPUESTA:
1. Analiza cuidadosamente el contexto proporcionado de los documentos oficiales.
2. Identifica las secciones, artículos o normas relevantes que aplican al caso.
3. Proporciona una respuesta estructurada que incluya:
   - Análisis del problema o consulta
   - Fundamento legal/normativo (cita los artículos o secciones específicas)
   - Recomendación o resolución clara
   - Explicación de cómo un experto humano llegaría a esta conclusión

4. Si la información no está en los documentos, indícalo claramente.
5. Mantén un tono profesional y académico.

RESPUESTA:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def inicializar_sistema(self):
        """
        Inicializa todo el sistema RAG: carga docs, crea vectorstore y configura QA
        """
        # Cargar documentos
        documentos = self.cargar_documentos()
        
        if not documentos:
            raise ValueError("No se encontraron documentos en la carpeta especificada")
        
        # Dividir en chunks
        chunks = self.dividir_documentos(documentos)
        
        # Crear vectorstore
        self.crear_vectorstore(chunks)
        
        # Crear el prompt template
        prompt = self.crear_prompt_template()
        
        # Configurar la cadena de QA
        print("Configurando cadena de preguntas y respuestas...")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 4}  # Recuperar los 4 fragmentos más relevantes
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        print("Sistema RAG inicializado correctamente!")
    
    def consultar(self, pregunta: str) -> dict:
        """
        Realiza una consulta al sistema RAG
        
        Args:
            pregunta: La consulta del usuario
            
        Returns:
            dict con 'respuesta' y 'fuentes'
        """
        if not self.qa_chain:
            raise ValueError("El sistema no ha sido inicializado. Ejecuta inicializar_sistema() primero.")
        
        print("\nProcesando consulta...")
        resultado = self.qa_chain.invoke({"query": pregunta})
        
        return {
            "respuesta": resultado["result"],
            "fuentes": resultado["source_documents"]
        }


# Función auxiliar para mostrar resultados
def mostrar_resultado(resultado: dict):
    """
    Muestra el resultado de forma formateada
    """
    print("\n" + "="*80)
    print("RESPUESTA DEL SISTEMA:")
    print("="*80)
    print(resultado["respuesta"])
    
    print("\n" + "="*80)
    print("FUENTES CONSULTADAS:")
    print("="*80)
    for i, doc in enumerate(resultado["fuentes"], 1):
        print(f"\n[Fuente {i}]")
        print(f"Documento: {doc.metadata.get('source', 'Desconocido')}")
        print(f"Página: {doc.metadata.get('page', 'N/A')}")
        print(f"Fragmento: {doc.page_content[:200]}...")


# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar el sistema
    rag = RAGSystemUNAH(
        documentos_path="./documentos",
        modelo="llama3.1"
    )
    
    # Cargar documentos y crear base de datos
    rag.inicializar_sistema()
    
    # Ejemplo de consulta
    pregunta = """
    Un estudiante ha reprobado el mismo curso tres veces. 
    ¿Qué establece el reglamento académico de la UNAH sobre esta situación?
    ¿Qué opciones tiene el estudiante?
    """
    
    resultado = rag.consultar(pregunta)
    mostrar_resultado(resultado)