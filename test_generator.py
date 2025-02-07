#import ollama
#from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.vectorstores import FAISS  
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains import RetrievalQA, StuffDocumentsChain
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import Language
from langchain_core.runnables import RunnablePassthrough

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

def setup_ai_model(model_name, prompt_file_path, pdf_file_path=None, code_file_path=None, question=None):
   
    if pdf_file_path is not None:
        documents = parse_pdf(pdf_file_path)
        embeddings = HuggingFaceEmbeddings() 
        vector_store = FAISS.from_documents(documents, embeddings)  

    elif code_file_path is not None:
        texts = parse_code(code_file_path)
        embeddings = OllamaEmbeddings(model="llama2:7b")
        vector_store = FAISS.from_documents(texts, embeddings) 

    # Connect retriever  
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) 
    document_variable_name = "context"
    
    llm = Ollama(model=model_name)
    
    # Craft the prompt template  
    prompt = open(prompt_file_path, 'r').read()
       
    document_prompt = PromptTemplate(  
        template=prompt,  
        input_variables=["context", "question"]  
    )  

    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | document_prompt
        | llm
        | StrOutputParser()
    )

    response = qa_chain.invoke(question)
    print(response)
    
# generate unit tests in C
#setup_ai_model('deepseek-coder:6.7b', code_file_path='sample_code\\rsu_client.c', prompt_file_path='prompts\\unit_test_c_prompt.txt', question='the function rsu_client_list_slot_attribute')

# generate test plan from PDF
setup_ai_model('deepseek-r1:latest', pdf_file_path='sample_pdf\\rsu.pdf', prompt_file_path='prompts\\test_plan_generation_prompt.txt', question='Programming Flash Memory with the Initial Remote System Update Image')
