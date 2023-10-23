from qdrant_client import QdrantClient
from vertexai.language_models import TextEmbeddingModel, TextGenerationModel

import constant


EMBEDDING_MODEL: TextEmbeddingModel = TextEmbeddingModel.from_pretrained('textembedding-gecko')
TEXT_MODEL: TextGenerationModel = TextGenerationModel.from_pretrained('text-bison')
QDRANT_CLIENT: QdrantClient = QdrantClient(location=constant.QDRANT_URL, prefer_grpc=True)
NB_DOC_CONTEXT: int = 3


def get_question() -> str:
    print('What is your question?')
    return input()


def get_context(question: str) -> list[str]:
    # Embed the question
    embeddings = EMBEDDING_MODEL.get_embeddings([question])[0].values

    # Retrieve docs
    qdrant_response = QDRANT_CLIENT.search(
        collection_name=constant.COLLECTION_NAME,
        query_vector=embeddings,
        limit=NB_DOC_CONTEXT,
        with_payload=True,
        with_vectors=False
    )

    return [doc.payload['response'] for doc in qdrant_response]


def get_prompt(question: str, context: list[str]) -> str:
    context = '\n'.join(context)

    return f"""SYSTEM: You are an intelligent assistant helping the users with their questions.
            
        Strictly Use ONLY the following pieces of context to answer the question at the end. Think step-by-step and then answer.
            
        Do not try to make up an answer:
         - If the answer to the question cannot be determined from the context alone, say "I cannot determine the answer to that."
         - If the context is empty, just say "I do not know the answerQ to that.
         "
        
        =============
        Context: {context}
        =============
        
        Question: {question}
        Helpful Answer:"""


def infer():
    while True:
        # Get question from user
        question: str = get_question()

        # Get context from our knowledge base to help LLM respond to the question
        context: list[str] = get_context(question=question)

        # Build the prompt with the context
        prompt: str = get_prompt(question=question, context=context)

        # Ask the response to the LLM
        response: str = TEXT_MODEL.predict(
            prompt=prompt,
            temperature=0,
            top_k=1
        ).text
        print(response)


if __name__ == '__main__':
    infer()