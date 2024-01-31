from llama_cpp import Llama
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import ElasticSearchBM25Retriever
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager
from langchain.llms.llamacpp import LlamaCpp
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.embeddings import HuggingFaceInstructEmbeddings


system_prompt = """You are a helpful assistant, you will use the provided context to answer user questions.
Read the given context before answering questions and think step by step. If you can not answer a user question based on 
the provided context, inform the user. Do not use any other information for answering user. Provide a detailed answer to the question."""


def get_prompt():
      B_INST, E_INST = "[INST]", "[/INST]"
      B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
      SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS

      instruction = """
      Context: {context}
      User: {question}"""

      prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
      prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
      return prompt
 
embeddings = HuggingFaceInstructEmbeddings(model_name='moka-ai/m3e-large', model_kwargs={"device": "cuda"})
elasticsearch_url = "http://124.220.46.48:9200/"
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
db = ElasticVectorSearch(index_name="test", embedding=embeddings, elasticsearch_url="http://localhost:9200")

retriever = db.as_retriever()

llm = LlamaCpp(
      model_path="/home/ubuntu/opt/text-generation-webui/models/yi-34b-chat.Q5_K_S.gguf",
      n_gpu_layers=-1,
      n_batch=512,
      n_ctx=2048
)
while True:

      qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={
                "prompt": get_prompt(),
            }
            
        )
      
      question = input("Question: ")
      res = qa(question)
      print(res)
