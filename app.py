"""
Interfaz web con Streamlit para el sistema RAG de la UNAH
"""

import streamlit as st
from rag_system import RAGSystemUNAH
import os

# Configuración de la página
st.set_page_config(
    page_title="Consultor UNAH - Sistema RAG",
    page_icon="🎓",
    layout="wide"
)

# Título y descripción
st.title("🎓 Sistema de Consulta de Documentos Oficiales UNAH")
st.markdown("""
Este sistema utiliza **Retrieval-Augmented Generation (RAG)** con **LLaMA 3.1** 
para responder consultas basadas en documentos oficiales de la Universidad Nacional 
Autónoma de Honduras.
""")

# Inicializar el sistema en session_state
@st.cache_resource
def inicializar_rag():
    """Inicializa el sistema RAG (solo se ejecuta una vez)"""
    with st.spinner("Cargando documentos y configurando el sistema..."):
        rag = RAGSystemUNAH(
            documentos_path="./documentos",
            modelo="llama3.1"
        )
        rag.inicializar_sistema()
    return rag

# Sidebar con información
with st.sidebar:
    st.header("⚙️ Configuración")
    
    st.markdown("### Modelo utilizado")
    st.info("🤖 LLaMA 3.1 (Ollama)")
    
    st.markdown("### Documentos cargados")
    docs_path = "./documentos"
    if os.path.exists(docs_path):
        archivos = [f for f in os.listdir(docs_path) if f.endswith(('.pdf', '.txt'))]
        st.success(f"📄 {len(archivos)} documentos encontrados")
        with st.expander("Ver lista de documentos"):
            for archivo in archivos:
                st.write(f"- {archivo}")
    else:
        st.error("❌ Carpeta de documentos no encontrada")
    
    st.markdown("---")
    st.markdown("### 📖 Ejemplos de consultas")
    st.markdown("""
    - ¿Qué dice el reglamento sobre la asistencia a clases?
    - ¿Cuál es el proceso para solicitar una reposición?
    - ¿Qué normas éticas deben seguir los docentes?
    - Un estudiante reprobó 3 veces el mismo curso, ¿qué procede?
    """)

# Área principal
try:
    # Inicializar RAG
    rag = inicializar_rag()
    st.success("✅ Sistema inicializado correctamente")
    
    # Tabs para diferentes modos
    tab1, tab2 = st.tabs(["💬 Consulta Simple", "🔍 Consulta Detallada"])
    
    with tab1:
        st.markdown("### Realiza tu consulta")
        
        consulta = st.text_area(
            "Escribe tu pregunta sobre normativas o reglamentos de la UNAH:",
            height=150,
            placeholder="Ejemplo: ¿Qué establece el reglamento sobre el plagio académico?"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            buscar = st.button("🔍 Consultar", type="primary", use_container_width=True)
        with col2:
            limpiar = st.button("🗑️ Limpiar", use_container_width=True)
        
        if limpiar:
            st.rerun()
        
        if buscar and consulta:
            with st.spinner("Analizando documentos y generando respuesta..."):
                resultado = rag.consultar(consulta)
                
                # Mostrar respuesta
                st.markdown("### 📋 Respuesta del Sistema")
                st.markdown(resultado["respuesta"])
                
                # Mostrar fuentes en un expander
                with st.expander("📚 Ver fuentes consultadas"):
                    for i, doc in enumerate(resultado["fuentes"], 1):
                        st.markdown(f"**Fuente {i}**")
                        st.markdown(f"- **Documento:** {doc.metadata.get('source', 'Desconocido')}")
                        st.markdown(f"- **Página:** {doc.metadata.get('page', 'N/A')}")
                        st.text_area(
                            f"Fragmento {i}",
                            doc.page_content[:300] + "...",
                            height=100,
                            disabled=True,
                            key=f"fuente_{i}"
                        )
                        st.markdown("---")
    
    with tab2:
        st.markdown("### Consulta con análisis de caso")
        
        st.markdown("#### 📝 Describe el caso o situación")
        contexto_caso = st.text_area(
            "Contexto del problema:",
            height=100,
            placeholder="Ejemplo: Un estudiante de ingeniería..."
        )
        
        situacion = st.text_area(
            "Situación específica:",
            height=100,
            placeholder="Ejemplo: Ha sido acusado de copiar en un examen..."
        )
        
        consulta_especifica = st.text_input(
            "Pregunta específica:",
            placeholder="¿Qué sanciones contempla el reglamento?"
        )
        
        if st.button("🔍 Analizar Caso", type="primary"):
            if contexto_caso and situacion and consulta_especifica:
                # Construir pregunta completa
                pregunta_completa = f"""
                CONTEXTO DEL CASO:
                {contexto_caso}
                
                SITUACIÓN:
                {situacion}
                
                CONSULTA:
                {consulta_especifica}
                
                Por favor, proporciona un análisis detallado incluyendo:
                1. Normativa aplicable
                2. Análisis de la situación
                3. Posibles resoluciones o recomendaciones
                """
                
                with st.spinner("Analizando el caso en detalle..."):
                    resultado = rag.consultar(pregunta_completa)
                    
                    # Mostrar respuesta en formato estructurado
                    st.markdown("### 📊 Análisis del Caso")
                    st.markdown(resultado["respuesta"])
                    
                    # Fuentes
                    with st.expander("📚 Documentos consultados"):
                        for i, doc in enumerate(resultado["fuentes"], 1):
                            st.markdown(f"**[{i}]** {doc.metadata.get('source', 'Desconocido')} - Pág. {doc.metadata.get('page', 'N/A')}")
            else:
                st.warning("⚠️ Por favor completa todos los campos")

except Exception as e:
    st.error(f"❌ Error al inicializar el sistema: {str(e)}")
    st.markdown("""
    ### Posibles soluciones:
    1. Verifica que Ollama esté ejecutándose: `ollama serve`
    2. Verifica que el modelo esté instalado: `ollama pull llama3.1`
    3. Verifica que la carpeta `./documentos` exista y contenga archivos PDF o TXT
    4. Revisa los logs de error arriba para más detalles
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Sistema RAG - UNAH | Powered by LLaMA 3.1 & LangChain</p>
</div>
""", unsafe_allow_html=True)