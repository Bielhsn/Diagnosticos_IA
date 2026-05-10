import json
import os
import warnings
import textwrap
from transformers import logging
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

class MedicalAssistantRAG:
    def __init__(self):
        # 1. Configurar Embeddings
        self.embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_db = None

        # Carrega o modelo de linguagem (LLM)
        print("🔄 Carregando modelo de IA (pode demorar na 1ª vez)...")
        self.llm = self._load_llm()

    def _load_llm(self):
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

        # Configuração da Pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=80,  # reduz
            temperature=0.1,
            do_sample=False,
            return_full_text=False
        )
        return HuggingFacePipeline(pipeline=pipe)

    def load_database(self, json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Erro: Arquivo não encontrado em {json_path}")
            return

        documents = []
        for item in data:
            doc = Document(
                page_content=item['conteudo'],
                metadata={"source": item['titulo'], "id": item['id']}
            )
            documents.append(doc)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        # Cria/Atualiza o banco vetorial
        self.vector_db = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding_function,
            persist_directory="./chroma_db"
        )
        print("✅ Base de conhecimento carregada.")

    def get_response(self, query):
        if not self.vector_db:
            return {"resposta": "Erro: Base de dados não carregada."}

        # 1. Retrieval (Busca)
        docs = self.vector_db.similarity_search(query, k=2)

        if not docs:
            return {"resposta": "Não encontrei informações nos protocolos."}

        context_text = "\n".join([f"- {d.page_content}" for d in docs])
        sources = list(set([d.metadata['source'] for d in docs]))  # Remove duplicadas

        # 2. Guardrails (Segurança)
        if "prescrever" in query.lower() or "receitar" in query.lower():
            return {
                "resposta": "⚠️ ALERTA DE SEGURANÇA: Como IA, não posso prescrever medicamentos. Consulte o protocolo e valide com um médico.",
                "fontes": sources
            }

        # 3. Generation (Prompt Otimizado para TinyLlama)
        prompt_template = PromptTemplate.from_template(
            """Você é um assistente médico hospitalar.
    
            REGRAS:
            - Responda curto e objetivo
            - Use apenas o contexto
            - Máximo 5 linhas
    
            Contexto:
            {context}
    
            Pergunta:
            {question}
    
            Resposta:"""
        )
        prompt = prompt_template.format(context=context_text, question=query)

        # Gera a resposta
        try:
            raw_response = self.llm.invoke(prompt)

            if isinstance(raw_response, dict):
                response = raw_response.get("text") or raw_response.get("generated_text") or str(raw_response)
            else:
                response = str(raw_response)
        except Exception as e:
            response = f"Erro na geração: {str(e)}"

        return {
            "resposta": response.strip(),  # Remove espaços extras
            "fontes": sources
        }