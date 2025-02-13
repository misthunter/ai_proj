#import ollama
#from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.vectorstores import FAISS  
from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains import RetrievalQA, StuffDocumentsChain
from langchain.retrievers.multi_vector import MultiVectorRetriever

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import Language
from langchain_core.runnables import RunnablePassthrough


def scrape_website(urls):
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    return (docs_transformed)

def parse_pdf(file_path):
    loader = PDFPlumberLoader(file_path)  
    docs = loader.load()
    return (docs)
    
def parse_code(file_path):
    code_doc = []

    with open(file_path, "r") as f:
        content = f.read()
        doc = Document(page_content=content, metadata={"filename": file_path, "file_index": 0})
        code_doc.append(doc)

    text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.C, chunk_size=1000, chunk_overlap=0
    )
    texts = text_splitter.split_documents(code_doc)

    return (texts)

def setup_ai_model(model_name, prompt_file_path, pdf_file_path=None, code_file_path=None, urls=None, question=None):
   
    if pdf_file_path is not None:
        documents = parse_pdf(pdf_file_path)
        embeddings = HuggingFaceEmbeddings() 
        vector_store = FAISS.from_documents(documents, embeddings)  
        pdf_retriever = vector_store.as_retriever(search_kwargs={"k": 3}) 

    if code_file_path is not None:
        texts = parse_code(code_file_path)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = FAISS.from_documents(texts, embeddings) 
        code_retriever = vector_store.as_retriever(search_kwargs={"k": 3}) 
        
    if urls is not None:
        texts = scrape_website(urls)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = FAISS.from_documents(texts, embeddings) 
        url_retriever = vector_store.as_retriever(search_kwargs={"k": 3}) 

    # Connect retriever  
    
    #document_variable_name = "context"
    
    llm = Ollama(model=model_name)
    
    # Craft the prompt template  
    prompt = open(prompt_file_path, 'r').read()
       
    document_prompt = PromptTemplate(  
        template=prompt,  
        input_variables=["code_context", "url_context", "question"]  
    )  

    qa_chain = (
        {"code_context": code_retriever, "url_context": url_retriever, "question": RunnablePassthrough()}
        | document_prompt
        | llm
        | StrOutputParser()
    )

    response = qa_chain.invoke(question)
    print(response)
    
## generate unit tests in C
#setup_ai_model('deepseek-coder:6.7b', code_file_path='sample_code\\rsu_client.c', prompt_file_path='prompts\\unit_test_c_prompt.txt', question='the function rsu_client_list_slot_attribute')

## generate test plan from PDF
#setup_ai_model('deepseek-r1:latest', pdf_file_path='sample_pdf\\rsu.pdf', prompt_file_path='prompts\\test_plan_generation_prompt.txt', question='Programming Flash Memory with the Initial Remote System Update Image')

## write a test case base on a description
#setup_ai_model('qwen2.5-coder', code_file_path='sample_code\\rsu_client.c', urls='https://altera-fpga.github.io/rel-24.2/embedded-designs/agilex-7/f-series/soc/rsu/ug-rsu-agx7f-soc/', prompt_file_path='prompts\\rsu_study_code.txt', question='Write a test case to display the max retry status')
#setup_ai_model('qwen2.5-coder', code_file_path='sample_code\\rsu_client.c', urls='https://altera-fpga.github.io/rel-24.2/embedded-designs/agilex-7/f-series/soc/rsu/ug-rsu-agx7f-soc/', prompt_file_path='prompts\\rsu_study_code.txt', question='Write a test case to erase slot 2 and add application2.rpd to slot 2')

##  writing tests step by step
#setup_ai_model('qwen2.5-coder', code_file_path='sample_code\\rsu.c', urls='https://altera-fpga.github.io/rel-24.2/embedded-designs/agilex-7/f-series/soc/rsu/ug-rsu-agx7f-soc/', prompt_file_path='prompts\\rsu_study_code.txt', question='Write a test case based on these steps: \n1) check dcmf version \n2) check the slot count \n3) get the slot info for slot 0 \n4) erase slot 1 \n5) load application2.rpd from mmc to ram \n6) program slot 1 with content of application2.rpd in ram buffer')

#docs = scrape_website("")
#print(docs)