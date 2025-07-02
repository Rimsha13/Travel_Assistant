import streamlit as st

from dotenv import load_dotenv
load_dotenv()
# ───── API Key ───── #
import os
openai_api_key = os.getenv("OPENAI_API_KEY")


from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter



# Use TextLoader with encoding explicitly set to 'utf-8'
loader = DirectoryLoader(
    "travel_data",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)
docs = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()




# ───── Streamlit Page Config ───── #
st.set_page_config(page_title="PlanMate", layout="centered")

# ───── Animated Tech Background Styling ───── #
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');

        /* Global background and font */
        .stApp {
            background-color: black;
            color: white;
            font-family: 'VT323', monospace;
        }

        /* PlanMate Title */
        .bubble-title {
            font-family: 'VT323', monospace;
            font-size: 96px;
            color: white;
            text-align: center;
            padding: 60px 0 20px 0;
            text-shadow: 2px 2px 8px #333;
            letter-spacing: 2px;
        }

        /* Chat message styling */
        .chat-message {
            background-color: #3a3a3a; /* grey bubble */
            color: white;
            font-family: 'VT323', monospace;
            font-size: 20px;
            padding: 14px 18px;
            border-radius: 12px;
            margin: 10px 0;
            max-width: 80%;
        }

        .user-message {
            margin-left: auto;
            text-align: right;
        }

        .bot-message {
            margin-right: auto;
            text-align: left;
        }

        /* Hide Reset Button */
        .stButton {
            display: none;
        }
    </style>

    <!-- Clean White Title -->
    <div class="bubble-title">PlanMate</div>
""", unsafe_allow_html=True)

# ───── Branding ───── #
st.markdown("\tYour futuristic AI travel assistant. Ask me anything about destinations, flights, visas & planning.")

# ───── LangChain Setup ───── #
if "memory" not in st.session_state:
    # st.session_state.memory = ConversationBufferMemory(return_messages=True)
    st.session_state.memory = ConversationBufferMemory(
    memory_key="chat_history",  # match the input key you're passing
    return_messages=True
)

    # RAG-based conversation
    st.session_state.conversation = ConversationalRetrievalChain.from_llm(
        llm=OpenAI(temperature=0.7),
        retriever=retriever,
        memory=st.session_state.memory
    )

    st.session_state.chat_history = []


# ───── Show Chat History ───── #
for speaker, msg in st.session_state.chat_history:
    align_class = "user-message" if speaker == "You" else "bot-message"
    with st.chat_message("user" if speaker == "You" else "assistant"):
        st.markdown(f'<div class="chat-message {align_class}">{msg}</div>', unsafe_allow_html=True)

# ───── Chat Input ───── #
user_input = st.chat_input("Type your travel query...")

if user_input:
    with st.chat_message("user"):
        st.markdown(f'<div class="chat-message user-message">{user_input}</div>', unsafe_allow_html=True)

    response = st.session_state.conversation.run({
    "question": user_input,
    "chat_history": st.session_state.chat_history
})



    with st.chat_message("assistant"):
        st.markdown(f'<div class="chat-message bot-message">{response}</div>', unsafe_allow_html=True)

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# # ───── Reset Button ───── #
# if st.button("🔄 Reset PlanMate"):
#     st.session_state.chat_history = []
#     st.session_state.memory = ConversationBufferMemory(return_messages=True)
#     st.session_state.conversation = ConversationChain(
#         llm=OpenAI(temperature=0.7),
#         memory=st.session_state.memory
#     )
#     st.success("PlanMate has been reset!")
