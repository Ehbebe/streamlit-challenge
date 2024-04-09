import re
from langchain.document_loaders import SitemapLoader
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.runnable import RunnableLambda
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
import streamlit as st


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)

with st.sidebar:
    st.divider()
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""

    api_key_input = st.empty()

    def reset_api_key():
        st.session_state["api_key"] = ""
        print(st.session_state["api_key"])

    if st.button(":red[Clear]"):
        reset_api_key()

    api_key = api_key_input.text_input(
        "**:blue[Please enter your OpenAI API Key.]**",
        value=st.session_state["api_key"],
        key="api_key_input",
    )

    if api_key != st.session_state["api_key"]:
        st.session_state["api_key"] = api_key
        st.rerun()

    url = st.text_input(
        "**:blue[Write down a URL]**",
        placeholder="https://example.com",
        value="https://developers.cloudflare.com/sitemap.xml",
    )

    url_name = url.split("://")[1].replace("/", "_") if url else None

    st.divider()
    st.markdown(
        """
        [GitHub Link](https://github.com/Ehbebe/fullstack-gpt/blob/main/pages/04_SiteGPT.py)
        """
    )


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm_for_backstage = ChatOpenAI(
    temperature=0.1,
    api_key=api_key if api_key else "_",
)


llm = ChatOpenAI(
    temperature=0.1,
    api_key=api_key if api_key else "_",
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ")


@st.cache_data(persist="disk", show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        filter_urls=(
            [
                r"https:\/\/developers.cloudflare.com/ai-gateway.*",
                r"https:\/\/developers.cloudflare.com/vectorize.*",
                r"https:\/\/developers.cloudflare.com/workers-ai.*",
            ]
            if "developers.cloudflare.com" in url_name
            else None
        ),
        parsing_function=parse_page,
    )
    loader.requests_per_second = 5
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(persist="disk", show_spinner="Embedding docs...")
def embeded_docs(_docs, url_name):
    cache_dir = LocalFileStore(f"./.cache/sitegpt/embeddings/{url_name}")

    embeddings = OpenAIEmbeddings(
        api_key=api_key,
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vector_store = FAISS.from_documents(_docs, cached_embeddings)
    retriever = vector_store.as_retriever()

    return retriever


answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                
    Examples:
                                                
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                
    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm_for_backstage
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {"question": question, "context": doc.page_content}
    #     )
    #     answers.append(result.content)
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\n\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


memory = ConversationBufferMemory(
    llm=llm_for_backstage,
    max_token_limit=1000,
    return_messages=True,
    memory_key="chat_history",
)


def load_memory(_):
    return memory.load_memory_variables({})["chat_history"]


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


if not api_key:
    st.warning("Please provide an **:blue[OpenAI API Key]** on the sidebar.")

if not url:
    st.warning("Please write down a **:blue[Sitemap URL]** on the sidebar.")


if api_key and url:
    if ".xml" not in url:
        st.warning("Please write down a Sitemap URL(**:blue[ .xml]**).")
    else:
        try:
            docs = load_website(url)
            docs_box = st.empty()
            retriever = embeded_docs(docs, url_name)
            send_message("I'm ready! Ask away!", "ai", save=False)
            paint_history()

            message = st.chat_input("Ask a question to the website.")
            if message:
                send_message(message, "human")

                chain = (
                    {
                        "docs": retriever,
                        "question": RunnablePassthrough(),
                    }
                    | RunnableLambda(get_answers)
                    | RunnableLambda(choose_answer)
                )

                def invoke_chain(question):
                    result = chain.invoke(question)
                    memory.save_context(
                        {"input": question},
                        {"output": result.content},
                    )
                    return result

                with st.chat_message("ai"):
                    invoke_chain(message)

        except Exception as e:
            e_str = str(e).lower()
            match = re.search(r"(api)(_|-|\s)(key)", e_str)
            if match:
                st.error("Please check the API_KEY.")

            st.expander("Error Details", expanded=True).write(f"Error: {e}")
            docs_box.write(docs)
