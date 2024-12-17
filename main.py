import boto3
import json
import streamlit as st
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Prompt Template
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.

<context>
{context}
</context>

Question: {question}

Assistant:"""

# Bedrock Client
bedrock = boto3.client(service_name="bedrock-runtime", region_name="eu-central-1")

# Get embedding model
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Load and split PDF documents
def get_documents():
    loader = PyPDFDirectoryLoader("pdf-data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    docs = text_splitter.split_documents(documents)
    return docs

# Vector Store FAISS
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

# Custom Bedrock LLM Class
def get_llm():
    class CustomBedrockLLM(Bedrock):
        def _prepare_input_and_invoke(self, prompt: str, **kwargs):
            payload = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 512,
                    "temperature": 0.7,
                    "topP": 1.0
                }
            }
            response = self.client.invoke_model(
                body=json.dumps(payload),
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
            )
            # Read and decode the response body
            raw_response = response["body"].read().decode("utf-8")
            parsed_response = json.loads(raw_response)

            # Return text and an empty dictionary as required
            return parsed_response.get("results", [])[0].get("outputText", ""), {}

    # Instantiate the custom Bedrock model
    llm = CustomBedrockLLM(
        model_id="amazon.titan-text-lite-v1",
        client=bedrock,
    )
    return llm

# Prompt Template for Retrieval QA
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Function to Query LLM and FAISS VectorStore
def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

# Streamlit App
def main():
    st.set_page_config("RAG Demo")
    st.header("RAG-Driven RainAI")
    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update or Create Vector Store:")

        # Create or Update Vector Store
        if st.button("Store Vector"):
            with st.spinner("Processing..."):
                docs = get_documents()
                get_vector_store(docs)
                st.success("Vector store created/updated successfully!")

        # Get Answer
        if st.button("Send"):
            with st.spinner("Processing..."):
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                llm = get_llm()
                answer = get_response_llm(llm, faiss_index, user_question)
                st.write(answer)

if __name__ == "__main__":
    main()
