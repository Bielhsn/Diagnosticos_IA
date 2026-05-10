# --- AJUSTE DE IMPORTS ---
# Agora importamos do arquivo 'generator' e chamamos a função 'gerar_dados_hospitalares'
from src.medical_assistant.generator import gerar_dados_hospitalares
from src.medical_assistant.rag_engine import MedicalAssistantRAG
import os
import textwrap

def main():
    print("--- INICIANDO SISTEMA DE ASSISTENTE MÉDICO (FASE 3) ---")

    # Definindo o caminho esperado (igual ao padrão do seu generator.py: data/fase3)
    caminho_dados = os.path.join("data", "fase3", "protocols.json")

    # 1. Garantir que os dados existem
    if not os.path.exists(caminho_dados):
        print(f"Gerando dados sintéticos...")
        # Chamada da função com o nome correto em português
        gerar_dados_hospitalares()

    # 2. Inicializar o Motor RAG
    print(f"Carregando base de dados de: {caminho_dados}")
    assistant = MedicalAssistantRAG()

    # O RAG vai tentar carregar o arquivo. Se der erro, ele avisa.
    assistant.load_database(caminho_dados)

    # 3. Loop de Interação
    print("\n✅ Assistente Pronto. Digite 'sair' para encerrar.")
    print("Exemplos de perguntas: 'Qual o protocolo de sepse?', 'Paciente com dor no peito', 'Prescrever Vancomicina'")

    while True:
        query = input("\n👨‍⚕️ Médico: ")
        if query.lower() in ['sair', 'exit']:
            break

        result = assistant.get_response(query)

        print("\n🤖 Assistente IA:\n")
        print(textwrap.fill(result['resposta'], width=100))

        # Verifica e imprime fontes se existirem
        if 'fontes' in result and result['fontes']:
            print(f"\n📚 Fontes Oficiais: {result['fontes']}")
        elif 'fonte' in result and result['fonte']:
            print(f"\n📚 Fontes Oficiais: {result['fonte']}")

        print("-" * 50)


if __name__ == "__main__":
    main()