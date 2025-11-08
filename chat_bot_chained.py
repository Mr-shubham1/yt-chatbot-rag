from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint,HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import chroma,FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import streamlit as st;
import re
from dotenv import load_dotenv




load_dotenv();

st.set_page_config(page_title="YouTube Video Chatbot", page_icon="🤖", layout="wide")
st.title("YouTube Video Chatbot")
st.caption("Ask questions about any YouTube video using AI-powered transcript analysis.")

if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

llmendpoint = HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-120b',
    task='text-generation'
)

open_source_model = ChatHuggingFace(llm=llmendpoint);
embedding_model = HuggingFaceEmbeddings();


# video_id = "pBRSZBtirAk" ;

st.markdown("### Enter YouTube Video URL or ID")
video_url = st.text_input(
    "Paste your YouTube link below",
    placeholder="e.g. https://youtu.be/pBRSZBtirAk",
    label_visibility="collapsed" 
)

def extractVideoId(url_or_id: str):
    if not url_or_id:
        return None
    if len(url_or_id.strip()) == 11 and " " not in url_or_id:
        return url_or_id.strip()
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url_or_id)
    return match.group(1) if match else None




if video_url:
    video_id = extractVideoId(video_url);
    if video_id:
        try:
            if st.session_state.transcript is None or st.session_state.get("last_video_id") != video_id:
                with st.spinner("Fetching transcript of the video..."):
                    yt_api = YouTubeTranscriptApi()
                    transcript_list = yt_api.fetch(video_id=video_id, languages=['en', 'hi'])
                    st.session_state.transcript = " ".join(chunk.text for chunk in transcript_list)
                    st.session_state.last_video_id = video_id
            st.success("✅ Transcript fetched successfully!")
        except TranscriptsDisabled:
            st.error("transcript not found");
            st.session_state.transcript = None;
        except Exception as e:
            st.error(e);
            st.session_state.transcript = None;
    else:
        st.session_state.transcript = None;
        st.warning("Invalid YouTube URL or video ID. Please check and try again.")
    if st.session_state.transcript:
        with st.spinner("chunking transcript and storing it in vector_store"):
            if st.session_state.vector_store is None or st.session_state.get('last_video_id') != video_id:
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200);
                chunks = splitter.create_documents([st.session_state.transcript])
                st.session_state.vector_store = FAISS.from_documents(chunks,embedding_model);
            retriever = st.session_state.vector_store.as_retriever(search_type='similarity',search_kwargs={'k':5});
            st.success("✅ Transcript Chunked and stored in vectorstore successfully!");
            st.markdown("### Ask your question about the video")


            input_query = st.text_area(
                "Type your question here",
                placeholder="e.g. What is the main topic discussed in this video?",
                height=100,
                label_visibility="collapsed"
            )
            retrieved_docs = retriever.invoke(input_query)
            def format_retrieved_docs(retrieved_docs):
                context_text = '\n\n'.join(doc.page_content for doc in retrieved_docs);
                return context_text;


            template = PromptTemplate(
                template="""
                You are a friendly and engaging AI who chats naturally with the user.
                Use the following transcript context to answer their question in a clear and human-like way.
    
                - Keep your tone conversational, like you are talking to a friend.
                - Use simple, natural sentences — avoid robotic replies.
                - If the context does not have enough info, say something polite like:
                "Hmm, I am not sure about that — the video did not mention it."
    
                Transcript Context:
                {context}
    
                  User’s Question:
                {query}
                """,
                input_variables=['context', 'query']
            )
            # model = ChatOpenAI(model='gpt-4o-mini',temperature=0.2);
            str_parser = StrOutputParser();

            chain1 = template | open_source_model | str_parser;
            chain2 = RunnableParallel({
            'context': retriever | RunnableLambda(format_retrieved_docs),
            'query': RunnablePassthrough()
            }) 
            chain3 = chain2 | chain1;
        
        if input_query:
            with st.spinner("Thinking, Please wait..."):
                llm_response  = chain3.invoke(input_query);
            st.subheader("Answer")
            st.write(llm_response);
        else:
            st.info("please enter a query");
else:
    st.info("Please enter Youtube video url to start conversation with the video")


