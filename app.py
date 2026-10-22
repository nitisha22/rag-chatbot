
import streamlit as st

from rag_pipeline import answer_question


st.set_page_config(
    page_title="University of Delhi RAG Chatbot",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 University of Delhi RAG Chatbot")
st.caption(
    "Ask questions about the M.Sc. Informatics brochure."
)

# Keep the conversation while the app is running
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View retrieved sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"**PDF page {source['page_number']}** "
                        f"(similarity: {source['score']:.4f})"
                    )
                    st.caption(source["text"])

# Accept a new question
question = st.chat_input(
    "Ask about eligibility, admission, or the programme..."
)

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the brochure and generating an answer..."):
            try:
                answer, sources = answer_question(question)
                st.markdown(answer)

                with st.expander("View retrieved sources"):
                    for source in sources:
                        st.markdown(
                            f"**PDF page {source['page_number']}** "
                            f"(similarity: {source['score']:.4f})"
                        )
                        st.caption(source["text"])

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as error:
                st.error(f"Something went wrong: {error}")
