"""
Script de prueba para el sistema RAG por consola
Útil para pruebas rápidas sin necesidad de la interfaz web
"""

from rag_system import RAGSystemUNAH, mostrar_resultado
import sys


def menu_principal():
    """Muestra el menú principal"""
    print("\n" + "="*80)
    print(" SISTEMA DE CONSULTA DE DOCUMENTOS OFICIALES UNAH ".center(80, "="))
    print("="*80)
    print("\nOpciones:")
    print("1. Realizar consulta simple")
    print("2. Analizar caso complejo")
    print("3. Salir")
    print("\nSelecciona una opción: ", end="")


def consulta_simple(rag: RAGSystemUNAH):
    """Modo de consulta simple"""
    print("\n" + "-"*80)
    print("MODO: CONSULTA SIMPLE")
    print("-"*80)
    print("\nIngresa tu consulta (o 'volver' para regresar):")
    
    consulta = input("> ").strip()
    
    if consulta.lower() == 'volver':
        return
    
    if not consulta:
        print("⚠️ La consulta no puede estar vacía")
        return
    
    try:
        resultado = rag.consultar(consulta)
        mostrar_resultado(resultado)
        
        input("\nPresiona Enter para continuar...")
    except Exception as e:
        print(f"\n❌ Error al procesar la consulta: {e}")


def analizar_caso(rag: RAGSystemUNAH):
    """Modo de análisis de caso complejo"""
    print("\n" + "-"*80)
    print("MODO: ANÁLISIS DE CASO COMPLEJO")
    print("-"*80)
    
    print("\n1. Describe el contexto del caso:")
    contexto = input("> ").strip()
    
    if not contexto:
        print("⚠️ El contexto no puede estar vacío")
        return
    
    print("\n2. Describe la situación específica:")
    situacion = input("> ").strip()
    
    if not situacion:
        print("⚠️ La situación no puede estar vacía")
        return
    
    print("\n3. ¿Qué deseas consultar?")
    consulta = input("> ").strip()
    
    if not consulta:
        print("⚠️ La consulta no puede estar vacía")
        return
    
    # Construir pregunta completa
    pregunta_completa = f"""
    CONTEXTO DEL CASO:
    {contexto}
    
    SITUACIÓN ESPECÍFICA:
    {situacion}
    
    CONSULTA:
    {consulta}
    
    Proporciona un análisis detallado que incluya:
    1. Identificación de las normativas aplicables
    2. Análisis de la situación conforme a los reglamentos
    3. Recomendaciones o resolución del caso
    4. Justificación desde la perspectiva de un experto
    """
    
    try:
        print("\n⏳ Analizando el caso...")
        resultado = rag.consultar(pregunta_completa)
        mostrar_resultado(resultado)
        
        input("\nPresiona Enter para continuar...")
    except Exception as e:
        print(f"\n❌ Error al procesar el caso: {e}")


def casos_ejemplo():
    """Retorna una lista de casos de ejemplo"""
    return [
        {
            "titulo": "Caso 1: Estudiante con múltiples reprobaciones",
            "consulta": """
            Un estudiante ha reprobado la asignatura de Cálculo I en tres ocasiones consecutivas.
            ¿Qué establece el reglamento académico de la UNAH sobre esta situación?
            ¿Qué opciones tiene el estudiante para continuar sus estudios?
            """
        },
        {
            "titulo": "Caso 2: Plagio académico",
            "consulta": """
            Un docente detectó que un estudiante copió gran parte de su trabajo de investigación
            de internet sin citar las fuentes. ¿Qué sanciones contempla el reglamento?
            ¿Cuál es el proceso disciplinario que debe seguirse?
            """
        },
        {
            "titulo": "Caso 3: Reposición de examen",
            "consulta": """
            Una estudiante no pudo asistir al examen final debido a una emergencia médica
            debidamente comprobada. ¿Tiene derecho a una reposición? ¿Cuál es el procedimiento?
            """
        }
    ]


def mostrar_casos_ejemplo(rag: RAGSystemUNAH):
    """Muestra y permite ejecutar casos de ejemplo"""
    casos = casos_ejemplo()
    
    print("\n" + "-"*80)
    print("CASOS DE EJEMPLO")
    print("-"*80)
    
    for i, caso in enumerate(casos, 1):
        print(f"\n{i}. {caso['titulo']}")
    
    print("\n0. Volver al menú principal")
    print("\nSelecciona un caso para analizar: ", end="")
    
    try:
        opcion = int(input().strip())
        
        if opcion == 0:
            return
        
        if 1 <= opcion <= len(casos):
            caso_seleccionado = casos[opcion - 1]
            print(f"\n📋 Analizando: {caso_seleccionado['titulo']}")
            print(f"\nConsulta:\n{caso_seleccionado['consulta']}")
            
            input("\nPresiona Enter para proceder con el análisis...")
            
            resultado = rag.consultar(caso_seleccionado['consulta'])
            mostrar_resultado(resultado)
            
            input("\nPresiona Enter para continuar...")
        else:
            print("⚠️ Opción no válida")
    except ValueError:
        print("⚠️ Por favor ingresa un número válido")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Función principal"""
    print("\n🎓 INICIANDO SISTEMA RAG - UNAH")
    print("Conectando con Ollama y cargando documentos...")
    
    try:
        # Inicializar el sistema
        rag = RAGSystemUNAH(
            documentos_path="./documentos",
            modelo="llama3.1"
        )
        
        rag.inicializar_sistema()
        
        print("\n✅ Sistema inicializado correctamente!")
        
        # Loop principal
        while True:
            menu_principal()
            
            try:
                opcion = input().strip()
                
                if opcion == "1":
                    consulta_simple(rag)
                elif opcion == "2":
                    analizar_caso(rag)
                elif opcion == "3":
                    print("\n👋 ¡Hasta luego!")
                    sys.exit(0)
                elif opcion == "ejemplos":  # Easter egg
                    mostrar_casos_ejemplo(rag)
                else:
                    print("\n⚠️ Opción no válida. Por favor selecciona 1, 2 o 3")
                    input("Presiona Enter para continuar...")
            
            except KeyboardInterrupt:
                print("\n\n👋 Sistema interrumpido. ¡Hasta luego!")
                sys.exit(0)
            except Exception as e:
                print(f"\n❌ Error inesperado: {e}")
                input("Presiona Enter para continuar...")
    
    except Exception as e:
        print(f"\n❌ Error al inicializar el sistema: {e}")
        print("\nVerifica que:")
        print("1. Ollama esté ejecutándose: ollama serve")
        print("2. El modelo esté descargado: ollama pull llama3.1")
        print("3. La carpeta './documentos' exista y contenga archivos")
        sys.exit(1)


if __name__ == "__main__":
    main()