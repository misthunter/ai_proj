#import ollama
#from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.vectorstores import FAISS  
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains import RetrievalQA, StuffDocumentsChain
#class test_generator:
#    def __init__(self, ):



def parse_source():
    return 0
    
def parse_pdf(file_path):
    loader = PDFPlumberLoader(file_path)  
    docs = loader.load()
    return (docs)

def setup_ai_model(model_name, file_path):
    embeddings = HuggingFaceEmbeddings()  
    documents = parse_pdf(file_path)
    vector_store = FAISS.from_documents(documents, embeddings)  

    # Connect retriever  
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) 
    document_variable_name = "context"
    
    llm = Ollama(model=model_name)
    
    # Craft the prompt template  
    prompt = """  
    1. Use ONLY the context below.  
    2. If unsure, say "I don’t know".  
    3. Keep answers under 4 sentences.  

    Context: {context}  

    Question: {question}  

    Answer:  
    """  
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)
    
    # Chain 1: Generate answers  
    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)  

    # Chain 2: Combine document chunks  
    document_prompt = PromptTemplate(  
        template="Context:\ncontent:{page_content}\nsource:{source}",  
        input_variables=["page_content", "source"]  
    )  

    # Final RAG pipeline  
    qa = RetrievalQA(  
        combine_documents_chain=StuffDocumentsChain(  
            llm_chain=llm_chain,  
            document_prompt=document_prompt,
            document_variable_name=document_variable_name
        ),  
        retriever=retriever  
    )
    
    response = qa({"query": "what is SDM?"})
    print(response)
    
#test_generator()
#parse_pdf()    
setup_ai_model('deepseek-coder:6.7b', 'sample_pdf\\rsu.pdf')