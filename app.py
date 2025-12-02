"""
Interfaz web mejorada con Streamlit para el sistema RAG de la UNAH
Incluye análisis de casos complejos y visualización de razonamiento experto
"""

import streamlit as st
from rag_system import RAGSystemUNAH
import os
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Sistema Experto UNAH - RAG",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .caso-complejo {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .fuente-documento {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Título y descripción
st.title("🎓 Sistema Experto de Consulta Normativa UNAH")
st.markdown("""
**Sistema RAG Avanzado** que emula el razonamiento de un experto normativo universitario  
Utiliza **LLaMA 3.1** con técnicas de *Retrieval-Augmented Generation* para analizar documentos oficiales
""")

# Inicializar el sistema en session_state
@st.cache_resource
def inicializar_rag():
    """Inicializa el sistema RAG (solo se ejecuta una vez)"""
    with st.spinner("🔄 Cargando documentos y configurando el sistema experto..."):
        rag = RAGSystemUNAH(
            documentos_path="./documentos",
            modelo="llama3.1"
        )
        rag.inicializar_sistema()
    return rag

# Función para cargar casos predefinidos
def cargar_casos_ejemplo():
    """Retorna casos de ejemplo para pruebas"""
    return {
        "Caso 1: Triple Reprobación": {
            "contexto": "Juan Pérez es estudiante de segundo año de Medicina. Tiene buen rendimiento general (índice 78%) excepto en Bioquímica.",
            "actores": "- Juan Pérez (estudiante)\n- Dra. Ana Martínez (docente)\n- Coordinación de Medicina",
            "situacion": "Juan ha reprobado Bioquímica tres veces consecutivas con calificaciones de 52%, 58% y 59%. Argumenta dificultades personales no documentadas médicamente.",
            "consulta": "¿Puede Juan matricular Bioquímica una cuarta vez? ¿Qué opciones tiene para continuar su carrera?",
            "aspectos": "Considerar: límites de reprobación, cambios de carrera, excepciones documentadas"
        },
        "Caso 2: Plagio Académico": {
            "contexto": "Andrea López presentó un trabajo de investigación en el curso de Metodología de la Investigación.",
            "actores": "- Andrea López (estudiante)\n- Dr. Carlos Gómez (docente)\n- Comisión de Ética Académica",
            "situacion": "El Dr. Gómez detectó que el 60% del trabajo fue copiado de internet sin citas. Andrea admite el error pero argumenta desconocimiento de las normas de citación APA.",
            "consulta": "¿Qué sanciones aplican? ¿Existe diferencia entre plagio intencional y por desconocimiento? ¿Qué proceso disciplinario debe seguirse?",
            "aspectos": "Considerar: gravedad del plagio, atenuantes, debido proceso, proporcionalidad"
        },
        "Caso 3: Reposición por Emergencia": {
            "contexto": "Luis Rodríguez cursaba Cálculo Integral con buen rendimiento (90% en parciales).",
            "actores": "- Luis Rodríguez (estudiante)\n- Ing. Patricia Flores (docente)\n- Coordinación Académica",
            "situacion": "Luis no pudo presentarse al examen final debido a hospitalización de emergencia por apendicitis. Tiene certificado médico del Hospital Escuela. El curso ya cerró hace 3 días.",
            "consulta": "¿Tiene derecho a reposición? ¿Cuál es el procedimiento? ¿Hay plazo límite para solicitar reposición con causa médica justificada?",
            "aspectos": "Considerar: causas de fuerza mayor, plazos de solicitud, derechos del estudiante"
        },
        "Caso 4: Conflicto Ético Docente": {
            "contexto": "El Dr. Fernández es profesor titular con 15 años de antigüedad. Tiene excelente historial académico.",
            "actores": "- Dr. Mario Fernández (docente)\n- Estudiantes del curso (15 personas)\n- Dirección de Carrera",
            "situacion": "Estudiantes denuncian que el Dr. Fernández hace comentarios inapropiados sobre apariencia física y hace preguntas personales incómodas. No hay evidencia de acoso físico. Hay testimonios escritos de 8 estudiantes.",
            "consulta": "¿Qué normas éticas aplican? ¿Es esto causal de sanción? ¿Qué proceso debe seguirse? ¿Qué protección tienen los estudiantes denunciantes?",
            "aspectos": "Considerar: código de ética docente, confidencialidad, debido proceso, protección a víctimas"
        }
    }

# Sidebar con información
with st.sidebar:
    st.header("⚙️ Configuración del Sistema")
    
    st.markdown("### 🤖 Modelo de IA")
    st.info("**LLaMA 3.1** (Ollama)")
    st.caption("Temperatura: 0.2 (precisión alta)")
    st.caption("Contexto: 4096 tokens")
    
    st.markdown("### 📚 Base de Conocimiento")
    docs_path = "./documentos"
    if os.path.exists(docs_path):
        archivos = [f for f in os.listdir(docs_path) if f.endswith(('.pdf', '.txt'))]
        st.success(f"✅ {len(archivos)} documentos cargados")
        with st.expander("📄 Ver documentos"):
            for archivo in archivos:
                st.write(f"• {archivo}")
    else:
        st.error("❌ Carpeta de documentos no encontrada")
    
    st.markdown("---")
    
    st.markdown("### 🎯 Características")
    st.markdown("""
    - ✅ Razonamiento experto
    - ✅ Citas normativas precisas
    - ✅ Análisis de casos complejos
    - ✅ Interpretación analógica
    - ✅ Trazabilidad completa
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Tips de uso")
    st.info("""
    **Consultas simples**: Preguntas directas sobre normativa
    
    **Casos complejos**: Situaciones que requieren análisis detallado con múltiples factores
    """)

# Área principal
try:
    # Inicializar RAG
    rag = inicializar_rag()
    st.success("✅ Sistema experto inicializado y listo")
    
    # Tabs para diferentes modos
    tab1, tab2, tab3 = st.tabs([
        "💬 Consulta Simple", 
        "⚖️ Análisis de Caso Complejo",
        "📚 Casos de Ejemplo"
    ])
    
    # TAB 1: Consulta Simple
    with tab1:
        st.markdown("### Realiza una consulta normativa")
        st.markdown("Ideal para preguntas directas sobre reglamentos, estatutos y normativas.")
        
        consulta = st.text_area(
            "**¿Cuál es tu consulta?**",
            height=150,
            placeholder="Ejemplo: ¿Qué establece el reglamento sobre asistencia mínima a clases para aprobar un curso?",
            help="Escribe tu pregunta de forma clara. El sistema buscará en todos los documentos oficiales."
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            buscar = st.button("🔍 Consultar", type="primary", use_container_width=True)
        with col2:
            limpiar = st.button("🗑️ Limpiar", use_container_width=True)
        
        if limpiar:
            st.rerun()
        
        if buscar and consulta:
            with st.spinner("🔄 Analizando documentos oficiales..."):
                resultado = rag.consultar(consulta)
                
                # Mostrar respuesta principal
                st.markdown("---")
                st.markdown("## 📋 Respuesta del Sistema Experto")
                st.markdown(resultado["respuesta"])
                
                # Mostrar fuentes
                st.markdown("---")
                st.markdown("## 📚 Documentos Consultados")
                
                for i, metadata in enumerate(resultado["fuentes_metadata"], 1):
                    with st.expander(f"📄 Fuente {i}: {os.path.basename(metadata['documento'])} - Pág. {metadata['pagina']}"):
                        st.markdown(f"**Documento completo:** `{metadata['documento']}`")
                        st.markdown(f"**Página:** {metadata['pagina']}")
                        st.markdown("**Fragmento relevante:**")
                        st.text_area(
                            f"Contenido {i}",
                            metadata['contenido'][:500] + "..." if len(metadata['contenido']) > 500 else metadata['contenido'],
                            height=150,
                            disabled=True,
                            key=f"fuente_simple_{i}",
                            label_visibility="collapsed"
                        )
                
                st.info(f"💡 Se consultaron **{resultado['numero_fuentes']} documentos** para generar esta respuesta")
    
    # TAB 2: Análisis de Caso Complejo
    with tab2:
        st.markdown("### Análisis Detallado de Caso")
        st.markdown("Para situaciones que requieren análisis profundo con múltiples factores y consideraciones.")
        
        st.markdown("---")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### 📝 Información del Caso")
            
            contexto_caso = st.text_area(
                "**Contexto general**",
                height=120,
                placeholder="Describe el contexto: quiénes son los involucrados, antecedentes relevantes, situación académica general...",
                help="Proporciona información de fondo que ayude a entender el caso"
            )
            
            actores = st.text_area(
                "**Actores involucrados**",
                height=100,
                placeholder="Lista las personas o entidades involucradas:\n- Estudiante: [nombre y características]\n- Docente: [nombre y rol]\n- Otras instancias...",
                help="Identifica claramente quiénes participan en la situación"
            )
        
        with col_right:
            st.markdown("#### ⚠️ Detalles del Problema")
            
            situacion = st.text_area(
                "**Situación específica**",
                height=120,
                placeholder="Describe los hechos concretos: qué ocurrió, cuándo, cómo, qué evidencia existe...",
                help="Sé específico sobre los eventos y circunstancias"
            )
            
            aspectos_adicionales = st.text_area(
                "**Aspectos a considerar**",
                height=100,
                placeholder="Menciona factores especiales: atenuantes, agravantes, precedentes, urgencia...",
                help="Factores que deberían considerarse en el análisis"
            )
        
        consulta_especifica = st.text_area(
            "**Preguntas específicas que necesitas resolver**",
            height=100,
            placeholder="1. ¿Qué normas aplican?\n2. ¿Qué opciones tiene el afectado?\n3. ¿Cuál es el procedimiento a seguir?",
            help="Lista las preguntas concretas que necesitas responder"
        )
        
        if st.button("⚖️ Analizar Caso Completo", type="primary", use_container_width=True):
            if all([contexto_caso, actores, situacion, consulta_especifica]):
                caso = {
                    "contexto": contexto_caso,
                    "actores": actores,
                    "situacion": situacion,
                    "consulta": consulta_especifica,
                    "aspectos_adicionales": aspectos_adicionales if aspectos_adicionales else "Análisis estándar"
                }
                
                with st.spinner("🔄 Realizando análisis experto profundo... (esto puede tardar 1-2 minutos)"):
                    resultado = rag.analizar_caso_complejo(caso)
                    
                    # Mostrar análisis
                    st.markdown("---")
                    st.markdown("## ⚖️ Análisis Experto del Caso")
                    st.markdown(resultado["respuesta"])
                    
                    # Documentos consultados
                    st.markdown("---")
                    st.markdown("## 📚 Base Documental del Análisis")
                    
                    for i, metadata in enumerate(resultado["fuentes_metadata"], 1):
                        with st.expander(f"📄 Documento {i}: {os.path.basename(metadata['documento'])}"):
                            st.markdown(f"**Fuente:** `{metadata['documento']}`")
                            st.markdown(f"**Página:** {metadata['pagina']}")
                            st.markdown(f"**Relevancia:** {metadata['relevancia']}")
                            st.markdown("---")
                            st.text_area(
                                "Contenido",
                                metadata['contenido'],
                                height=200,
                                disabled=True,
                                key=f"fuente_complejo_{i}",
                                label_visibility="collapsed"
                            )
                    
                    st.success(f"✅ Análisis completado consultando **{resultado['numero_fuentes']} documentos oficiales**")
                    
                    # Opción de descarga
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    contenido_descarga = f"""
ANÁLISIS DE CASO - SISTEMA EXPERTO UNAH
Generado: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{'='*80}
CASO ANALIZADO
{'='*80}

CONTEXTO:
{contexto_caso}

ACTORES:
{actores}

SITUACIÓN:
{situacion}

CONSULTA:
{consulta_especifica}

{'='*80}
ANÁLISIS EXPERTO
{'='*80}

{resultado["respuesta"]}

{'='*80}
DOCUMENTOS CONSULTADOS: {resultado['numero_fuentes']}
{'='*80}
"""
                    st.download_button(
                        label="📥 Descargar Análisis Completo",
                        data=contenido_descarga,
                        file_name=f"analisis_caso_{timestamp}.txt",
                        mime="text/plain"
                    )
            else:
                st.warning("⚠️ Por favor completa al menos: contexto, actores, situación y consulta específica")
    
    # TAB 3: Casos de Ejemplo
    with tab3:
        st.markdown("### 📚 Casos de Ejemplo Predefinidos")
        st.markdown("Casos reales o hipotéticos para demostrar las capacidades del sistema")
        
        casos = cargar_casos_ejemplo()
        
        # Selector de caso
        caso_seleccionado = st.selectbox(
            "**Selecciona un caso de ejemplo:**",
            options=list(casos.keys()),
            format_func=lambda x: x
        )
        
        if caso_seleccionado:
            caso = casos[caso_seleccionado]
            
            st.markdown("---")
            st.markdown(f"## {caso_seleccionado}")
            
            # Mostrar detalles del caso en columnas
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📋 Contexto**")
                st.info(caso["contexto"])
                
                st.markdown("**👥 Actores**")
                st.info(caso["actores"])
            
            with col2:
                st.markdown("**⚠️ Situación**")
                st.warning(caso["situacion"])
                
                st.markdown("**❓ Consulta**")
                st.error(caso["consulta"])
            
            st.markdown("**🔍 Aspectos a considerar**")
            st.caption(caso["aspectos"])
            
            if st.button(f"⚖️ Analizar: {caso_seleccionado}", type="primary", use_container_width=True):
                caso_dict = {
                    "contexto": caso["contexto"],
                    "actores": caso["actores"],
                    "situacion": caso["situacion"],
                    "consulta": caso["consulta"],
                    "aspectos_adicionales": caso["aspectos"]
                }
                
                with st.spinner("🔄 Analizando caso de ejemplo..."):
                    resultado = rag.analizar_caso_complejo(caso_dict)
                    
                    st.markdown("---")
                    st.markdown("## 📊 Resultado del Análisis")
                    st.markdown(resultado["respuesta"])
                    
                    with st.expander("📚 Ver documentos consultados"):
                        for i, metadata in enumerate(resultado["fuentes_metadata"], 1):
                            st.markdown(f"**[{i}]** {metadata['documento']} - Pág. {metadata['pagina']}")

except Exception as e:
    st.error(f"❌ Error al inicializar el sistema: {str(e)}")
    st.markdown("""
    ### 🔧 Soluciones posibles:
    
    1. **Verifica Ollama:**
       ```bash
       ollama serve
       ```
    
    2. **Verifica el modelo:**
       ```bash
       ollama pull llama3.1
       ```
    
    3. **Verifica los documentos:**
       - Carpeta `./documentos` debe existir
       - Debe contener archivos PDF o TXT
    
    4. **Revisa los logs** arriba para más detalles
    """)
    
    if st.button("🔄 Reintentar inicialización"):
        st.cache_resource.clear()
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Sistema RAG Avanzado - UNAH</strong></p>
    <p>Powered by LLaMA 3.1 • LangChain • ChromaDB • Streamlit</p>
    <p>Emulando razonamiento de experto normativo universitario</p>
</div>
""", unsafe_allow_html=True)