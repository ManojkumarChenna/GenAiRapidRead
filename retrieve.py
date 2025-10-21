import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

st.title("🌐 Web Article Q&A with Groq LLM")

# Initialize session state
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "loaded_urls" not in st.session_state:
    st.session_state.loaded_urls = []

# Get Groq API key from environment
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("⚠️ GROQ_API_KEY not found in .env file. Please add it and restart the app.")
    st.info("Create a `.env` file in your project root with:\n```\nGROQ_API_KEY=your_api_key_here\n```")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key


# Cache the LLM and embeddings to avoid reloading
@st.cache_resource
def initialize_models():
    model_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=500,
    )
    model_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return model_llm, model_embeddings


llm, embeddings = initialize_models()

# Sidebar: URL inputs
st.sidebar.header("📄 Enter Article URLs")
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

load_button = st.sidebar.button("🔄 Load Articles", use_container_width=True)

if load_button:
    urls = [url.strip() for url in [url1, url2, url3] if url.strip()]

    if not urls:
        st.sidebar.warning("⚠️ Please enter at least one URL.")
    else:
        with st.spinner("Loading articles and creating index..."):
            all_docs = []
            successful_urls = []
            failed_urls = []

            # Loop through URLs to load and split
            for url in urls:
                try:
                    loader = WebBaseLoader([url])
                    data = loader.load()

                    if not data:
                        failed_urls.append((url, "No content loaded"))
                        continue

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    docs = splitter.split_documents(data)

                    all_docs.extend(docs)
                    successful_urls.append(url)

                except Exception as e:
                    failed_urls.append((url, str(e)))

            # Create FAISS index if we have documents
            if all_docs:
                try:
                    vector_index = FAISS.from_documents(all_docs, embeddings)
                    st.session_state.vector_index = vector_index
                    st.session_state.loaded_urls = successful_urls

                    st.sidebar.success(f"✅ Successfully indexed {len(successful_urls)} article(s)!")

                    if failed_urls:
                        st.sidebar.warning("⚠️ Failed URLs:")
                        for url, error in failed_urls:
                            st.sidebar.text(f"• {url[:50]}...")
                            st.sidebar.caption(f"Error: {error[:100]}")
                except Exception as e:
                    st.sidebar.error(f"❌ Error creating index: {str(e)}")
            else:
                st.sidebar.error("❌ No documents were successfully loaded.")

# Display currently loaded articles
if st.session_state.loaded_urls:
    with st.sidebar.expander("📚 Currently Loaded Articles"):
        for i, url in enumerate(st.session_state.loaded_urls, 1):
            st.text(f"{i}. {url[:60]}...")

# Q&A Interface
if st.session_state.vector_index is not None:
    st.markdown("---")
    st.subheader("💬 Ask Questions About Your Articles")

    retriever = st.session_state.vector_index.as_retriever(
        search_kwargs={"k": 3}  # Return top 3 relevant chunks
    )

    # Create a RAG prompt template
    template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)


    # Create RAG chain using LCEL
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    query = st.text_input("Enter your question:", placeholder="What is this article about?")

    if query:
        with st.spinner("🔍 Fetching answer..."):
            try:
                # Get answer
                answer = rag_chain.invoke(query)

                # Get source documents separately
                source_docs = retriever.get_relevant_documents(query)

                # Display answer
                st.markdown("### Answer:")
                st.write(answer)

                # Show sources
                if source_docs:
                    with st.expander("📖 View Sources"):
                        for i, doc in enumerate(source_docs[:3], 1):
                            st.markdown(f"**Source {i}:**")
                            st.text(doc.page_content[:300] + "...")
                            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                                st.caption(f"From: {doc.metadata['source']}")
                            st.markdown("---")

            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
else:
    st.info("👆 Please load articles using the sidebar to start asking questions.")