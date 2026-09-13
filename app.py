import streamlit as st

from rag import retrieve, generate_answer


# ==========================================
# Page configuration
# ==========================================

st.set_page_config(
    page_title="EZITECH Customer Support RAG",
    page_icon="🤖",
    layout="wide"
)


# ==========================================
# Header
# ==========================================

st.title("🤖 EZITECH Customer Support RAG")

st.write(
    "Ask a question about customer support. "
    "The system searches a real customer-support "
    "ticket dataset and provides an answer using "
    "retrieved information."
)


# ==========================================
# Question
# ==========================================

question = st.text_input(
    "Enter your question",
    placeholder="Example: How can I reset my password?"
)


# ==========================================
# Search button
# ==========================================

if st.button("Search"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching support tickets..."
        ):

            results = retrieve(
                question,
                top_k=3
            )

            answer = generate_answer(
                question,
                results
            )


        # ==================================
        # Answer
        # ==================================

        st.subheader("Answer")

        st.info(answer)


        # ==================================
        # Retrieved evidence
        # ==================================

        st.subheader(
            "Retrieved Evidence"
        )

        st.success(
            f"Retrieved {len(results)} relevant support ticket chunks."
        )

        st.write(
            "These are the exact chunks retrieved "
            "from the customer-support dataset."
        )


        # ==================================
        # Show chunks
        # ==================================

        for number, result in enumerate(
            results,
            start=1
        ):

            with st.expander(
                f"Retrieved Chunk {number}"
            ):

                st.write(
                    f"**Similarity Score:** "
                    f"{result['score']:.4f}"
                )

                st.write(
                    f"**Ticket ID:** "
                    f"{result['ticket_id']}"
                )

                st.write(
                    f"**Chunk ID:** "
                    f"{result['chunk_id']}"
                )

                st.markdown(
                    "**Exact Retrieved Chunk:**"
                )

                st.code(
                    result["text"],
                    language="text"
                )