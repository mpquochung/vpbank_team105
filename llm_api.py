from langchain_openai import ChatOpenAI
from langchain_community.llms import Bedrock
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
import os

OPENAI_API_KEY = os.environ['OPENAI_API_KEY'] #hide this
def get_llm(streaming_callback = None, model: str = "gpt-3.5-turbo-0125", temperature = 0.1):
    if "gpt" in model:
        if streaming_callback:
            llm = ChatOpenAI(
                model = model, 
                temperature= temperature, 
                api_key=OPENAI_API_KEY, 
                callbacks=[streaming_callback],
                streaming=True)    
        else:
            llm = ChatOpenAI(
                model = model, 
                temperature= 0, 
                api_key=OPENAI_API_KEY)
    
    # Bedrock model, configure for each models, here use llama and anthropic models only        
    else:
        #llama model
        if "meta" in model:
            model_kwargs = {
            "max_gen_len": 1024,
            "temperature": temperature,
            "top_p": 0.01,
            }

            if streaming_callback:
                llm = Bedrock(
                    model_id= model,
                    model_kwargs=model_kwargs,
                    streaming=True,
                    callbacks=[streaming_callback],
                )
            else:
                llm = Bedrock(
                    model_id= model,
                    model_kwargs=model_kwargs,
                    streaming=False,
                )

        #anthropic model
        if "anthropic" in model:
            model_kwargs = {
            "max_tokens": 1024,
            "temperature": temperature, 
            "top_k": 250, 
            "top_p": 1, 
            }

            if streaming_callback:
                llm = BedrockChat(
                    model_id= model,
                    model_kwargs=model_kwargs,
                    streaming=True,
                    callbacks=[streaming_callback],
                )
            else:
                llm = BedrockChat(
                    model_id= model,
                    model_kwargs=model_kwargs,
                    streaming=False,
                )
    return llm

def get_embedding(text= None, model: str = "openai"):
    if "openai" in model:
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)  #create a Embeddings client

    else:
        embeddings = BedrockEmbeddings() 
    
    if text:
        return embeddings.embed_query(text)
    else:
        return embeddings