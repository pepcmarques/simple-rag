import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

from langchain_community.llms import LlamaCpp

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

verbose = False

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    llm = LlamaCpp(
      model_path="synthia-7b-v2.0-16k.Q4_K_M.gguf",
      # max tokens the model can account for when processing a response
      # make it large enough for the question and answer
      n_ctx=4096,
      # number of layers to offload to the GPU 
      # GPU is not strictly required but it does help
      n_gpu_layers=32,
      # number of tokens in the prompt that are fed into the model at a time
      n_batch=1024,
      # use half precision for key/value cache; set to True per langchain doc
      f16_kv=True,
      verbose=verbose,
    )

    response_text = llm.invoke(
        prompt,
        max_tokens=4096,
        temperature=0.2,
        # nucleus sampling (mass probability index)
        # controls the cumulative probability of the generated tokens
        # the higher top_p the more diversity in the output
        top_p=0.1
    )

    for doc, _score in results:
        print(doc.metadata.get("id", None), _score)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
