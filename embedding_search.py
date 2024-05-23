from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import file_loader
from llm_api import get_embedding

def get_index(cv_directory): #creates and returns an in-memory vector store to be used in the application
    
    embeddings = get_embedding(model = "bedrock")
    
    #loader = PyPDFLoader(file_path="test_data/cncnmai.pdf")
    loader = file_loader.load_docs(root_directory=cv_directory)

    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=embeddings,
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=10),
    )

    index_from_loader = index_creator.from_documents(loader)

    return index_from_loader

def get_similarity_search_results(index, question):
    results = index.vectorstore.similarity_search_with_score(question, k=20)
    flattened_results = [{"content":res[0].page_content,"score":res[1], "cv": res[0].metadata["source"]} for res in results] #flatten results for easier display and handling
    #"content":res[0].page_content
    return flattened_results

# def get_embedding(text):
#     embeddings = OpenAIEmbeddings(api_key=)  #create a Embeddings client
    
#     return embeddings.embed_query(text)

cv_directory = "test_data/"
vector_index = get_index(cv_directory = cv_directory)
response_content = get_similarity_search_results(index=vector_index, question="")
print(response_content)