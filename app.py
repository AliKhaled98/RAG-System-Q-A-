from flask import Flask, request, render_template
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

app = Flask(__name__)

# Initialize LangChain components
embedding = FastEmbedEmbeddings()
vector_store = Chroma(persist_directory="./soccer_chroma_db", embedding_function=embedding)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt = PromptTemplate.from_template(
    """
    <s> [Instructions] You are a friendly assistant. Answer the question based only on the following context.
    If you don't know the answer, then reply, No Context available for this question {input}. [/Instructions] </s>
    [Instructions] Question: {input}
    Context: {context}
    Answer: [/Instructions]
    """
)

model = ChatOllama(model="llama3")
document_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    sources = []
    query = ""
    
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            try:
                result = rag_chain.invoke({"input": query})
                answer = result["answer"]
                sources = [
                    {
                        "source": doc.metadata.get("source", "Unknown source"),
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                    }
                    for doc in result["context"]
                ]
            except Exception as e:
                answer = f"Error processing query: {str(e)}"
    
    return render_template("index.html", query=query, answer=answer, sources=sources)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)