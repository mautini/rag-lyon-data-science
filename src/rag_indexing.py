import json

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
from vertexai.language_models import TextEmbeddingModel

import constant

EMBEDDING_MODEL: TextEmbeddingModel = TextEmbeddingModel.from_pretrained('textembedding-gecko')
EMBEDDING_SIZE: int = 768
SIMILARITY_DISTANCE: str = 'Cosine'
QDRANT_CLIENT: QdrantClient = QdrantClient(location=constant.QDRANT_URL, prefer_grpc=True)
DATA_FILE: str = '../data/knowledge_base.json'


def index_data():
    QDRANT_CLIENT.recreate_collection(
        collection_name=constant.COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_SIZE,
            distance=SIMILARITY_DISTANCE
        )
    )

    with open(DATA_FILE, 'r') as f:
        knowledge_data = json.load(f)

    vectors: list[list[float]] = []
    payloads: list[dict[str, any]] = []

    for item in knowledge_data:
        vectors.append(
            EMBEDDING_MODEL.get_embeddings([item['question']])[0].values  # Call Vertex AI embedding model
        )
        payloads.append(
            {
                'question': item['question'],
                'response': item['response'],
                'source': item['source']
            }
        )

    # Upload collection handle batch insert
    QDRANT_CLIENT.upload_collection(
        collection_name=constant.COLLECTION_NAME,
        vectors=vectors,
        payload=payloads
    )


if __name__ == '__main__':
    index_data()
