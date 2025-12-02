"""
Sistema RAG Mejorado para consultar documentos oficiales de la UNAH
Utiliza LLaMA 3.1 a través de Ollama con técnicas avanzadas de RAG
"""

import os
from typing import List, Dict, Optional
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
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class RAGSystemUNAH:
    def __init__(self, documentos_path: str = "./documentos", modelo: str = "llama3.1"):
        """
        Inicializa el sistema RAG mejorado
        
        Args:
            documentos_path: Ruta a la carpeta con documentos de la UNAH
            modelo: Nombre del modelo en Ollama (llama3.1, mistral, gemma:2b)
        """
        self.documentos_path = documentos_path
        self.modelo_name = modelo
        self.vectorstore = None
        self.qa_chain = None
        
        # Configurar embeddings con modelo multilingüe optimizado
        print("Cargando modelo de embeddings multilingüe...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Configurar LLM con Ollama y streaming
        print(f"Conectando con Ollama - Modelo: {self.modelo_name}...")
        self.llm = Ollama(
            model=self.modelo_name,
            temperature=0.2,  # Temperatura baja para mayor precisión
            top_p=0.9,
            repeat_penalty=1.1,
            num_ctx=4096,  # Contexto más amplio
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
    
    def cargar_documentos(self) -> List:
        """
        Carga todos los documentos PDF y TXT de la carpeta especificada
        """
        print(f"Cargando documentos desde {self.documentos_path}...")
        
        documentos = []
        
        # Cargar PDFs con mejor manejo de errores
        try:
            pdf_loader = DirectoryLoader(
                self.documentos_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True,
                silent_errors=True
            )
            docs_pdf = pdf_loader.load()
            documentos.extend(docs_pdf)
            print(f"  ✓ {len(docs_pdf)} documentos PDF cargados")
        except Exception as e:
            print(f"  ⚠ Error cargando PDFs: {e}")
        
        # Cargar archivos de texto
        try:
            txt_loader = DirectoryLoader(
                self.documentos_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                show_progress=True,
                silent_errors=True
            )
            docs_txt = txt_loader.load()
            documentos.extend(docs_txt)
            print(f"  ✓ {len(docs_txt)} documentos TXT cargados")
        except Exception as e:
            print(f"  ⚠ Error cargando TXTs: {e}")
        
        print(f"Total de documentos cargados: {len(documentos)}")
        return documentos
    
    def dividir_documentos(self, documentos: List) -> List:
        """
        Divide los documentos en chunks optimizados para mejor recuperación
        """
        print("Dividiendo documentos en fragmentos optimizados...")
        
        # Splitter mejorado con separadores jerárquicos
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,  # Chunks más grandes para mejor contexto
            chunk_overlap=300,  # Mayor solapamiento para capturar contexto completo
            length_function=len,
            separators=[
                "\n\n\n",  # Secciones principales
                "\n\n",    # Párrafos
                "\n",      # Líneas
                ".",       # Oraciones
                " ",       # Palabras
                ""
            ],
            add_start_index=True  # Añadir índice de inicio para referencia
        )
        
        chunks = text_splitter.split_documents(documentos)
        print(f"Total de fragmentos creados: {len(chunks)}")
        return chunks
    
    def crear_vectorstore(self, chunks: List):
        """
        Crea la base de datos vectorial con ChromaDB con configuración mejorada
        """
        print("Creando base de datos vectorial optimizada...")
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db",
            collection_metadata={"hnsw:space": "cosine"}  # Usar similitud coseno
        )
        
        print("Base de datos vectorial creada exitosamente!")
    
    def crear_prompt_maestro(self) -> PromptTemplate:
        """
        Crea el prompt maestro mejorado que emula el razonamiento de un experto
        """
        template = """Eres el **Dr. Mauricio Hernández**, Asesor Normativo Principal de la Secretaría General de la Universidad Nacional Autónoma de Honduras (UNAH), con más de 25 años de experiencia interpretando normativas universitarias. Has participado en la redacción y reforma de múltiples reglamentos institucionales y eres reconocido por tu capacidad analítica y aplicación justa de las normas.

=== TU TAREA ===
Analizar la consulta o situación presentada aplicando el mismo razonamiento metódico que utilizaría un experto jurídico-administrativo universitario. Debes:

1. **Comprender el contexto completo**: Identificar todos los elementos relevantes del caso
2. **Localizar normativas aplicables**: Buscar artículos directos y normas relacionadas
3. **Razonar como experto**: Aplicar interpretación literal, sistemática y analógica según corresponda
4. **Justificar cada conclusión**: Explicar el "por qué" de cada parte de tu análisis
5. **Proporcionar soluciones prácticas**: Ofrecer caminos de acción claros y viables

=== METODOLOGÍA DE ANÁLISIS (como lo haría un experto) ===

**Paso 1: Deconstrucción del caso**
Antes de buscar normas, un experto identifica:
- ¿Quiénes son los actores involucrados? (estudiante, docente, autoridad)
- ¿Qué tipo de situación es? (académica, disciplinaria, administrativa, ética)
- ¿Qué derechos y obligaciones están en juego?
- ¿Hay conflicto de normas o vacíos legales?

**Paso 2: Búsqueda normativa estratificada**
Un experto busca en este orden:
1. Normas específicas que regulen exactamente el caso
2. Normas generales del mismo ámbito (si no hay específicas)
3. Principios generales del derecho universitario
4. Analogía con casos similares regulados
5. Jurisprudencia o precedentes institucionales (si están documentados)

**Paso 3: Interpretación contextualizada**
- **Literal**: ¿Qué dice exactamente el texto?
- **Sistemática**: ¿Cómo se relaciona con otras normas del mismo documento?
- **Teleológica**: ¿Cuál es el espíritu y finalidad de la norma?
- **Histórica**: ¿Por qué se creó esta norma? (si se conoce el contexto)

**Paso 4: Ponderación y resolución**
Cuando hay conflicto entre normas o derechos:
- Aplicar principio de especialidad (norma específica > norma general)
- Aplicar principio de jerarquía (estatuto > reglamento > normativa interna)
- Ponderar derechos en conflicto con proporcionalidad
- Favorecer interpretación que proteja derechos fundamentales del estudiante

=== DOCUMENTOS OFICIALES DISPONIBLES ===
{context}

=== CONSULTA O CASO PLANTEADO ===
{question}

=== FORMATO DE RESPUESTA (estructura de análisis experto) ===

**🔍 1. ANÁLISIS PRELIMINAR DEL CASO**
[Expón tu comprensión del caso como si se lo explicaras a un colega. Identifica: actores, naturaleza del problema, derechos en juego, complejidad del caso]

**📚 2. MARCO NORMATIVO APLICABLE**

*2.1 Normativa Directa*
[Cita textualmente los artículos que regulan específicamente este caso]
- **[Documento]** - Artículo X: "[cita textual]"
- [Explica por qué este artículo aplica directamente]

*2.2 Normativa Complementaria o Supletoria*
[Si no hay norma directa, identifica las más cercanas]
- **[Documento]** - Artículo Y: "[cita textual]"
- [Explica la relación analógica o supletoria]

*2.3 Principios Generales Aplicables*
[Menciona principios no escritos pero aplicables: debido proceso, buena fe, proporcionalidad, etc.]

**⚖️ 3. RAZONAMIENTO JURÍDICO-ADMINISTRATIVO**

*3.1 Interpretación de las normas*
[Analiza cómo un experto interpretaría cada artículo aplicable al caso concreto. Usa razonamiento literal, sistemático o teleológico según corresponda]

*3.2 Aplicación al caso concreto*
[Conecta la norma abstracta con los hechos específicos del caso. Muestra el razonamiento paso a paso]

*3.3 Consideraciones adicionales*
[Factores que un experto consideraría: precedentes, equidad, impacto en el estudiante, proporcionalidad de medidas]

**✅ 4. CONCLUSIONES Y RESOLUCIÓN**

*4.1 Respuesta directa a la consulta*
[Responde de forma clara y concisa qué es lo que procede según la normativa]

*4.2 Derechos del afectado*
[Enumera claramente qué derechos tiene la persona involucrada]

*4.3 Procedimiento a seguir*
[Paso a paso qué debe hacer el estudiante/docente/autoridad]
- Paso 1: [Acción concreta]
- Paso 2: [Siguiente acción]
- Plazos: [Si aplican]
- Instancias: [A dónde acudir]

*4.4 Escenarios posibles*
[Si hay múltiples desenlaces según decisiones o apelaciones]

**🎯 5. RECOMENDACIÓN EXPERTA**
[Como asesor experimentado, ¿qué aconsejarías? Incluye aspectos estratégicos, no solo normativos]

**📊 6. NIVEL DE CERTEZA Y TRAZABILIDAD**

*Nivel de certeza:*
- [ ] **ALTA CERTEZA** → Respuesta basada en normativa explícita y clara
- [ ] **CERTEZA MODERADA** → Respuesta basada en interpretación sistemática de normas relacionadas
- [ ] **CERTEZA BAJA** → Respuesta basada en analogía o principios generales
- [ ] **NO REGULADO** → Situación sin normativa aplicable identificada en los documentos

*Trazabilidad documental:*
- Documentos consultados: [Lista]
- Artículos citados: [Lista completa]
- Lagunas identificadas: [Si las hay]

*Recomendación de validación:*
[Si la certeza es baja o hay ambigüedad, sugiere consultar con: Secretaría General, Dirección de X, etc.]

---

**NOTAS METODOLÓGICAS IMPORTANTES:**

1. **Transparencia interpretativa**: Siempre explica SI estás interpretando, analogizando o aplicando literalmente
2. **Honestidad epistemológica**: Si no hay norma, dilo claramente. No inventes artículos
3. **Razonamiento visible**: Muestra el proceso mental, no solo el resultado
4. **Enfoque en el usuario**: Traduce lo jurídico a lenguaje accesible sin perder precisión
5. **Empatía institucional**: Entiende que las normas buscan proteger a la comunidad universitaria

Procede con tu análisis."""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def inicializar_sistema(self):
        """
        Inicializa todo el sistema RAG con configuración optimizada
        """
        # Cargar documentos
        documentos = self.cargar_documentos()
        
        if not documentos:
            raise ValueError("No se encontraron documentos en la carpeta especificada")
        
        # Dividir en chunks
        chunks = self.dividir_documentos(documentos)
        
        # Crear vectorstore
        self.crear_vectorstore(chunks)
        
        # Crear el prompt maestro mejorado
        prompt = self.crear_prompt_maestro()
        
        # Configurar la cadena de QA con retrieval mejorado
        print("Configurando cadena de preguntas y respuestas...")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance para diversidad
                search_kwargs={
                    "k": 6,  # Recuperar más documentos
                    "fetch_k": 20,  # Búsqueda inicial más amplia
                    "lambda_mult": 0.7  # Balance relevancia-diversidad
                }
            ),
            chain_type_kwargs={
                "prompt": prompt,
                "verbose": False
            },
            return_source_documents=True
        )
        
        print("✅ Sistema RAG inicializado correctamente!")
    
    def consultar(self, pregunta: str, modo: str = "normal") -> Dict:
        """
        Realiza una consulta al sistema RAG
        
        Args:
            pregunta: La consulta del usuario
            modo: 'normal' o 'detallado' (con fuentes expandidas)
            
        Returns:
            dict con 'respuesta', 'fuentes' y 'metadatos'
        """
        if not self.qa_chain:
            raise ValueError("El sistema no ha sido inicializado. Ejecuta inicializar_sistema() primero.")
        
        print("\n🔍 Procesando consulta...")
        resultado = self.qa_chain.invoke({"query": pregunta})
        
        # Preparar metadatos de las fuentes
        fuentes_metadata = []
        for doc in resultado["source_documents"]:
            fuentes_metadata.append({
                "documento": doc.metadata.get('source', 'Desconocido'),
                "pagina": doc.metadata.get('page', 'N/A'),
                "contenido": doc.page_content,
                "relevancia": "Alta"  # Podrías calcular score si usas similarity search
            })
        
        return {
            "respuesta": resultado["result"],
            "fuentes": resultado["source_documents"],
            "fuentes_metadata": fuentes_metadata,
            "numero_fuentes": len(resultado["source_documents"]),
            "modo": modo
        }
    
    def analizar_caso_complejo(self, caso: Dict[str, str]) -> Dict:
        """
        Método especializado para análisis de casos complejos
        
        Args:
            caso: Dict con keys 'contexto', 'situacion', 'actores', 'consulta'
        """
        # Construir pregunta estructurada
        pregunta_estructurada = f"""
=== CASO COMPLEJO PARA ANÁLISIS ===

**CONTEXTO GENERAL:**
{caso.get('contexto', 'No especificado')}

**ACTORES INVOLUCRADOS:**
{caso.get('actores', 'No especificado')}

**SITUACIÓN ESPECÍFICA:**
{caso.get('situacion', 'No especificado')}

**CONSULTA ESPECÍFICA:**
{caso.get('consulta', 'No especificado')}

**ASPECTOS A CONSIDERAR:**
{caso.get('aspectos_adicionales', 'Análisis estándar según metodología experta')}

Por favor, realiza un análisis completo siguiendo tu metodología de experto normativo, considerando todos los elementos proporcionados y las posibles implicaciones.
"""
        
        return self.consultar(pregunta_estructurada, modo="detallado")


# Función mejorada para mostrar resultados
def mostrar_resultado(resultado: Dict, modo: str = "completo"):
    """
    Muestra el resultado de forma formateada y profesional
    """
    print("\n" + "="*100)
    print(" RESPUESTA DEL SISTEMA EXPERTO ".center(100, "="))
    print("="*100)
    print(resultado["respuesta"])
    
    if modo == "completo":
        print("\n" + "="*100)
        print(" DOCUMENTOS CONSULTADOS ".center(100, "="))
        print("="*100)
        
        for i, metadata in enumerate(resultado["fuentes_metadata"], 1):
            print(f"\n📄 [FUENTE {i}]")
            print(f"   Documento: {metadata['documento']}")
            print(f"   Página: {metadata['pagina']}")
            print(f"   Relevancia: {metadata['relevancia']}")
            print(f"\n   Fragmento relevante:")
            print(f"   {'-'*90}")
            # Mostrar primeras 400 caracteres del fragmento
            fragmento = metadata['contenido'][:400]
            print(f"   {fragmento}{'...' if len(metadata['contenido']) > 400 else ''}")
            print(f"   {'-'*90}")
        
        print(f"\n💡 Total de fuentes consultadas: {resultado['numero_fuentes']}")


# Ejemplo de uso mejorado
if __name__ == "__main__":
    print("="*100)
    print(" SISTEMA RAG MEJORADO - UNAH ".center(100))
    print(" Emulando razonamiento de experto normativo ".center(100))
    print("="*100)
    
    # Inicializar el sistema
    rag = RAGSystemUNAH(
        documentos_path="./documentos",
        modelo="llama3.1"
    )
    
    # Cargar documentos y crear base de datos
    rag.inicializar_sistema()
    
    print("\n" + "="*100)
    print(" EJEMPLO 1: Consulta Simple ".center(100))
    print("="*100)
    
    # Ejemplo de consulta simple
    pregunta_simple = """
    ¿Qué establece el reglamento académico sobre el número máximo de veces 
    que un estudiante puede reprobar una misma asignatura?
    """
    
    resultado = rag.consultar(pregunta_simple)
    mostrar_resultado(resultado)
    
    print("\n\n" + "="*100)
    print(" EJEMPLO 2: Caso Complejo ".center(100))
    print("="*100)
    
    # Ejemplo de caso complejo
    caso_complejo = {
        "contexto": """
        María González es estudiante de tercer año de Ingeniería Civil en la UNAH.
        Tiene un índice académico de 75% y es alumna regular sin antecedentes disciplinarios.
        """,
        "actores": """
        - María González (estudiante)
        - Ing. Roberto Mejía (docente del curso)
        - Coordinación de Ingeniería Civil
        """,
        "situacion": """
        María reprobó el curso de "Análisis Estructural II" en dos ocasiones previas 
        con calificaciones de 55% y 58%. En su tercera matrícula del curso, obtuvo 59%, 
        quedando a solo 1% de aprobar. 
        
        María argumenta que en el examen final hubo un error en la suma de puntos, 
        y que debió obtener 61%. El docente revisó y confirma que la calificación 
        está correcta. Sin embargo, María presenta certificado médico indicando que 
        estuvo bajo tratamiento psicológico durante el período por ansiedad severa.
        """,
        "consulta": """
        1. ¿Puede María matricular nuevamente este curso?
        2. ¿El certificado médico es causal para una reconsideración?
        3. ¿Qué opciones tiene María según la normativa?
        4. ¿Existe algún recurso de apelación en casos límite como este?
        """,
        "aspectos_adicionales": """
        Considerar: situación académica general, precedentes de casos similares,
        proporcionalidad de medidas, debido proceso, y derechos del estudiante.
        """
    }
    
    resultado_complejo = rag.analizar_caso_complejo(caso_complejo)
    mostrar_resultado(resultado_complejo, modo="completo")