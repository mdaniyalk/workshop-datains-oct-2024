#!/usr/bin/env python
# coding: utf-8

# # Basic RAG
# 
# Retrieval-augmented generation (RAG) is an AI framework that synergizes the capabilities of LLMs and information retrieval systems. It’s useful to answer questions or generate content leveraging external knowledge. 
# 
# There are two main steps in RAG: 
# 1. retrieval: retrieve relevant information from a knowledge base with text embeddings stored in a vector store; 
# 2. generation: insert the relevant information to the prompt for the LLM to generate information. 
# 
# In this guide, we will walk through a very basic example of RAG with four implementations:
# 
# - RAG from scratch with Llama 3.2 (using free Groq API),  and Faiss
# - RAG with Llama 3.2 and LangChain
# 

# ## RAG from scratch
# 
# This section aims to guide you through the process of building a basic RAG from scratch.

# ### Setup and Installation

# Install the required libraries

# In[1]:


get_ipython().system('pip install numpy==1.26.4 faiss-cpu==1.8.0 openai==1.14.3 sentence-transformers pandas==1.5.3')


# Download the PubMedQA Labeled for demonstration purposes.

# In[2]:


get_ipython().system('wget https://raw.githubusercontent.com/pubmedqa/pubmedqa/refs/heads/master/data/ori_pqal.json -O pqa_labelled.json')


# ### Import Libraries

# In[3]:


import numpy as np
import faiss
import json
import pandas as pd

from sentence_transformers import SentenceTransformer
from openai import OpenAI

client = OpenAI(
    api_key="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", 
    base_url='https://api.groq.com/openai/v1',
)


# ### Load and view the dataset

# In[4]:


with open('pqa_labelled.json', 'r') as f:
    data = json.load(f)

transformed_data = {
    "questions": [],
    "contexts": [],
    "answers": []
}

for item in data.values():
    transformed_data['questions'].append(item['QUESTION'])
    transformed_data['contexts'].append(item['CONTEXTS'])
    transformed_data['answers'].append(item['LONG_ANSWER'])

df = pd.DataFrame(transformed_data)
df.head()


# ## Split document into chunks
# 
# In a RAG system, it is crucial to split the document into smaller chunks so that it’s more effective to identify and retrieve the most relevant information in the retrieval process later. In this example, we simply split our text by character, combine 2048 characters into each chunk.

# In[5]:


chunk_size = 2048
chunks = []

for text in df['contexts']:
    text = "\n".join(text)
    chunks += [text[i:i + chunk_size] for i in range(0, len(text), chunk_size) if len(text[i:i + chunk_size]) > 0]

print(f"Total chunks: {len(chunks)}")

chunks = chunks[:200]
print(f"For testing, we will use only {len(chunks)} chunks")


# In[6]:


def get_text_embedding(sentences):
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    embeddings = model.encode(sentences)
    return embeddings


# In[7]:


text_embeddings = get_text_embedding(chunks)


# In[8]:


text_embeddings.shape


# In[9]:


text_embeddings


# ### Load into a vector database
# Once we get the text embeddings, a common practice is to store them in a vector database for efficient processing and retrieval. There are several vector database to choose from. In our simple example, we are using an open-source vector database Faiss, which allows for efficient similarity search.  
# 
# With Faiss, we instantiate an instance of the Index class, which defines the indexing structure of the vector database. We then add the text embeddings to this indexing structure.
# 

# In[10]:


d = text_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(text_embeddings)


# ### Create embeddings for a question
# Whenever users ask a question, we also need to create embeddings for this question using the same embedding models as before.
# 

# In[11]:


question = "Explain what is Amblyopia?"
question_embeddings = get_text_embedding([question])
question_embeddings.shape


# In[12]:


question_embeddings


# ### Retrieve similar chunks from the vector database
# We can perform a search on the vector database with `index.search`, which takes two arguments: the first is the vector of the question embeddings, and the second is the number of similar vectors to retrieve. This function returns the distances and the indices of the most similar vectors to the question vector in the vector database. Then based on the returned indices, we can retrieve the actual relevant text chunks that correspond to those indices.
# 

# In[13]:


D, I = index.search(question_embeddings, k=3)
print(I)


# In[14]:


retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
print(retrieved_chunk)


# ### Combine context and question in a prompt and generate response
# 
# Finally, we can offer the retrieved text chunks as the context information within the prompt. Here is a prompt template where we can include both the retrieved text and user question in the prompt.
# 
# 

# In[15]:


def prompt_template(question, retrieved_chunk):
    prompt = f"""
    Answer the following question based only on the provided context.

    <context>
    {retrieved_chunk}
    </context>

    Question: {question}
    """
    return prompt


# In[16]:


def run_llm(user_message, model="llama-3.2-3b-preview"):
    system_message = "You are a helpful assistant. You are given a question and a context. You need to answer the question based on the context."
    messages = [{"role": "system", "content": system_message}]
    messages += [{"role": "user", "content": user_message}]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return completion.choices[0].message.content


# In[17]:


prompt = prompt_template(question, retrieved_chunk)

run_llm(prompt)


# ### Test the dataset

# In[18]:


data_id = 1

question, answers = df['questions'][data_id], df['answers'][data_id]

question_embeddings = get_text_embedding([question])
D, I = index.search(question_embeddings, k=3)
retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
prompt = prompt_template(question, retrieved_chunk)

response = run_llm(prompt)
print("Question:", question)
print("RAG Response:", response)
print("Ground Truth:", answers)


# ## LangChain

# In[19]:


get_ipython().system('pip install langchain==0.1.13 langchain-community==0.0.29 langchain-openai==0.1.1 langchain-huggingface==0.0.3')


# In[22]:


from langchain_core.documents import Document
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain


# In[23]:


# Load data
docs = [Document(page_content="\n".join(doc)) for doc in df['contexts']]
docs = docs[:200]


# In[24]:


# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)


# In[25]:


# Define the embedding model
embeddings = SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en-v1.5")
# Create the vector store
vector = FAISS.from_documents(documents, embeddings)
# Define a retriever interface
retriever = vector.as_retriever()


# In[26]:


# Define LLM
model = ChatOpenAI(
    model_name='llama-3.2-3b-preview',
    openai_api_key="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", 
    openai_api_base="https://api.groq.com/openai/v1"
)



# In[27]:


# Define prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}
""")


# In[28]:


# Create a retrieval chain to answer questions
document_chain = create_stuff_documents_chain(model, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# In[29]:


response = retrieval_chain.invoke({"input": "Explain what is Amblyopia?"})
print(response["answer"])


# In[31]:


data_id = 1

question, answers = df['questions'][data_id], df['answers'][data_id]


response = retrieval_chain.invoke({"input": question})
print("Question:", question)
print("RAG Response:", response["answer"])
print("Ground Truth:", answers)

