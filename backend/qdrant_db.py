from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models
import uuid

COLLECTION_NAME = "breast_cancer_cases"
VECTOR_SIZE = 1280  # EfficientNet-B0 output size

# ✅ LOCAL MODE (no server needed)
client = QdrantClient(path="qdrant_data", prefer_grpc=False)
def create_collection():
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        print(f"✅ Collection '{COLLECTION_NAME}' created.")
    else:
        print(f"⚡ Collection '{COLLECTION_NAME}' already exists.")

def store_case(embedding: list, label: str, image_path: str):
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding,
        payload={"label": label, "image_path": image_path}
    )
    client.upsert(collection_name=COLLECTION_NAME, points=[point])

def search_similar(embedding, top_k: int = 3):
    """Search for similar cases in Qdrant using query_points"""
    try:
        # Convert numpy array to list if needed
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        
        # Use query_points - the correct method for local/REST mode
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=top_k,
            with_payload=True
        )

        return [
            {
                "label": result.payload.get("label", "Unknown"),
                "image_path": result.payload.get("image_path", ""),
                "similarity_score": round(float(result.score), 3)
            }
            for result in results.points
        ]

    except AttributeError as e:
        print(f"⚠️ Qdrant method error: {e}")
        # Fallback: return empty list (search not critical to functionality)
        return []
    except Exception as e:
        print(f"⚠️ Qdrant search failed: {e}")
        import traceback
        traceback.print_exc()
        return []