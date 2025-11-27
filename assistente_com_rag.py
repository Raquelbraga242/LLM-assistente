import os
import tempfile
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title = "Assistente de Manuais", page_icon = ":100:", layout = "wide")

with st.sidebar:
    st.header("Configurações")
    api_key = st.text_input("GROQ API Key", type="password")
    st.divider()
    st.subheader("Como usar")
    st.write("1) Insira sua API Key\n2) Envie um manual técnico em PDF\n3) Faça perguntas.")
    st.info("A IA pode cometer erros. Sempre valide as informações apresentadas.")

st.title("Assistente Técnico baseado em RAG")
st.caption("Baseado no conteúdo do manual enviado pelo usuário")

if not api_key:
    st.warning("Informe a GROQ API Key para continuar.")
    st.stop()

os.environ["GROQ_API_KEY"] = api_key

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2, max_tokens=1024)

pdf_file = st.file_uploader(
    "Envie um manual técnico em PDF (equipamento, software, dispositivo, máquina etc.)",
    type=["pdf"]
)

if "chroma_dir" not in st.session_state:
    st.session_state.chroma_dir = tempfile.mkdtemp(prefix="chroma_rag_manual_")

def cria_banco(pdf_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    docs = PyPDFLoader(tmp_path).load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-bert-base-dot-v5")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=st.session_state.chroma_dir
    )

    return vectordb

if pdf_file and "vectordb_ready" not in st.session_state:
    with st.spinner("Indexando manual..."):
        st.session_state.vectordb = cria_banco(pdf_file.read())
        st.session_state.vectordb_ready = True
        st.success("Indexação concluída.")

retriever = None
if st.session_state.get("vectordb_ready"):
    retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 3})

# Prompt para contexto técnico
system_block = """
Você é um assistente técnico especializado em manuais de equipamentos, sistemas, dispositivos e máquinas.

Regras:
1. Responda somente com base no conteúdo do PDF enviado.
2. Se a resposta não estiver no manual, diga explicitamente: "Não encontrei essa informação no manual."
3. Formate a resposta em três seções obrigatórias:
   - Resumo técnico
   - Referências do manual (com trechos entre aspas e número de página)
   - Procedimentos sugeridos ou próximos passos
4. Se houver conflito entre o manual e conhecimento externo, priorize o manual e registre a divergência.
"""

qa_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_block),
    (
        "human",
        "Pergunta: {question}\n\n"
        "Conteúdo recuperado do manual:\n{context}\n\n"
        "Responda de forma objetiva, técnica e aplicável."
    )
])

def formata_docs(docs):
    out = []
    for d in docs:
        meta = d.metadata or {}
        page = meta.get("page", "?")
        out.append(f"[p.{page}] \"{d.page_content[:800]}{'…' if len(d.page_content) > 800 else ''}\"")
    return "\n\n".join(out)

if "chat_ready" not in st.session_state:
    st.session_state.chat_ready = True

pergunta = st.text_area(
    "Escreva sua dúvida técnica",
    height=120,
    placeholder="Exemplo: Onde está descrito o processo de calibração? Como reiniciar o sistema?"
)

col1, col2 = st.columns([1, 1])
with col1:
    btn = st.button("Consultar")

if btn:
    if not retriever:
        st.error("Envie um manual em PDF primeiro.")
        st.stop()

    rag_pipeline = (
        RunnableParallel(
            context=retriever | formata_docs,
            question=RunnablePassthrough()
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    with st.spinner("Consultando o manual..."):
        answer = rag_pipeline.invoke(pergunta)

    st.markdown("### Resposta")
    st.write(answer)
