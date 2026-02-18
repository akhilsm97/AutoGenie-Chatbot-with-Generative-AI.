# import torch
# import logging, warnings
# from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig, pipeline
# from peft import LoraConfig,get_peft_model,TaskType
# from datasets import load_dataset
# from langchain_community.document_loaders import TextLoader
# from langchain_community.llms import HuggingFacePipeline
# from langchain_community.document_loaders import JSONLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# import re


# # Loggong & Warning

# logging.basicConfig(level=logging.INFO)
# warnings.filterwarnings("ignore")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logging.info(f"{device} connected successfully!")

# bnb_Config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type='nf4',
#     bnb_4bit_compute_dtype=torch.float16
# )

# Model_name = 'fine_tuned_phi_2'
# try:
#     tokenizer = AutoTokenizer.from_pretrained(Model_name)
#     model = AutoModelForCausalLM.from_pretrained(Model_name,quantization_config=bnb_Config)
#     model.to(device)
#     model.eval()
#     logging.info(f"{Model_name} Connected Successfully!")
# except Exception as e:
#     logging.error(f"{Model_name} is failed to load!")    
#     raise SystemError("Exited due to model load error!")

# # Text generation pipeline (without streamer)
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     temperature=0.3,
#     max_new_tokens=300,
#     repetition_penalty=1.1
# )

# # Wrap it in LangChain interface
# llms = HuggingFacePipeline(pipeline=pipe)

# custom_prompt = PromptTemplate.from_template("""
# You are a knowledgeable and reliable **vehicle assistant**. 
# Always use the provided **context** to answer the question. 
# If the context does not contain enough information, say: 
# "I'm sorry, I couldn't find enough relevant medical information to answer that."

# Guidelines:
# - Be concise, clear, and medically accurate.
# - Do not guess or make up information outside the context.
# - If relevant, format the answer in short paragraphs or bullet points for readability.

# Context:
# {context}

# User's Question:
# {question}

# Final Answer:
# """)

# # Load and process JSON data
# loader = TextLoader("Vehicle_Types.txt")
# docs = loader.load()

# # Chunking
# splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
# chunks = splitter.split_documents(docs)

# # Embeddings + FAISS index
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vector_db = FAISS.from_documents(chunks, embedding_model)
# vector_db.save_local("FAISS_INDEX")

# # Create retriever
# retriever = vector_db.as_retriever()

# # Retrieval QA chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llms,
#     retriever=retriever,
#     return_source_documents=True,
#     chain_type="stuff",
#     chain_type_kwargs={"prompt": custom_prompt}
# )

# def generate_response(query):
#     result = qa_chain.invoke(query)
#     answer = result.get("result", "")  # extract the string safely
#     return answer

import logging
import warnings
import torch
import wikipedia
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

bnb_Config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model_name = "fine_tuned_phi_2"
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_Config,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.eval()

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

text_generation = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    streamer=streamer,
    top_k=50,
    top_p=0.9,
    temperature=0.7,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=text_generation)

# Load docs & embeddings
data_loader = TextLoader("Vehicle_Types.txt")
docs = data_loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)
retriever = vectorstore.as_retriever()

def build_prompt_with_context(query):
    results = retriever.get_relevant_documents(query)
    relevant = [doc for doc in results if query.lower() in doc.page_content.lower()]
    context = "\n\n".join([doc.page_content for doc in relevant[:2]]) if relevant else ""

    if not context.strip() or len(context.split()) < 20:
        try:
            wiki_summary = wikipedia.summary(query, sentences=3)
            context = f"(Fetched from Wikipedia)\n\n{wiki_summary}"
        except Exception:
            context = "(No reliable context found from local DB or Wikipedia)"

    return f"""You are a helpful medical assistant. 
Answer the user's question **only if the context is relevant**. 
If not, say "I'm sorry, I couldn't find enough relevant medical information."

Context:
{context}

User's Question: {query}
Answer:"""

def generate_response(query):
    print("MY Query:", query)
    prompt = build_prompt_with_context(query)
    response = llm.invoke(prompt)
    return response.split("Answer:")[-1].strip() if "Answer:" in response else response.strip()

