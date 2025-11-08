from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import chroma,FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv();

# step 1:- INDEXING -> process of creating knowledge base
# a -> load source knowledge (Document ingestion)
# b -> break them in chunks
# c -> generate embedding for each chunk
# d -> stote in any vector store

video_id = "pBRSZBtirAk" # Only ID not URL



# 1(a) START -> DOCUMENT INGESTION

try:
    # if you dont care about language , this will return the best one
    yt_api = YouTubeTranscriptApi();
    transcript_list = yt_api.fetch(video_id=video_id,languages=['en','hi']);
    # print(transcript_list);
    # flaten it to plain text
    transcript = " ".join(chunk.text for chunk in transcript_list)
    # print(transcript);

except TranscriptsDisabled:
    print("transcript not found");
except Exception as e:
    print("error: ",e);

# 1(a) DONE



# 1(b) START -> CHUNKING VIA RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200);
chunks = splitter.create_documents([transcript])
# print(len(chunks))
# print(chunks[0]);

# 1(b) DONE


# 1(c) and 1(d) START -> generate embedding for each chunk using an embedding model and store it in a vector store

embedding_model = OpenAIEmbeddings();

vector_store = FAISS.from_documents(chunks,embedding_model);

# print(vector_store.index_to_docstore_id);


# for key,value in vector_store.index_to_docstore_id.items():
#     print(key,value,vector_store.docstore.search(value))





# step 2:- Retrieval -> process of retrieving the most relevant chunks from knowledge base
# step 2-> creating Retriever and retrieving the most relavant docs from vectorstore
retriever = vector_store.as_retriever(search_type='similarity',search_kwargs={'k':5});

# retrivers are runnable
input_query = input("enter your question you want to know in this video")
retrieved_docs = retriever.invoke(input_query)
# print(retrieved_docs,len(retrieved_docs));

# generate context_text from retrieved_docs

context_text = '\n\n'.join(doc.page_content for doc in retrieved_docs);
# print(context_text);






# step 3 -> Augmentation :- prompt = querry + retrieved docs

template = PromptTemplate(
    template="""
    you are a helpful assistance, answer ONLY from the provided transcript context.
    If the the context is insufficient, just say you don't know.
    {context}
    question:{query} 
""",
    input_variables=['context','query']

);



# this is augmented prompt
prompt = template.invoke({"context":context_text,"query":input_query});

# print(prompt);





# step 4 :- generation

model = ChatOpenAI(model='gpt-4o-mini',temperature=0.2);

llm_response = model.invoke(prompt);

print(llm_response.content);
