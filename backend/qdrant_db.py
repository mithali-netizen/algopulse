import os
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION_NAME = "breast_cancer_cases"
VECTOR_SIZE = 1280  # EfficientNet-B0 output size

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if QDRANT_URL and QDRANT_API_KEY:
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=False
    )
    print(f"🌩️ Using Qdrant cloud at {QDRANT_URL}")
else:
    backend_dir = Path(__file__).resolve().parent
    local_path = backend_dir / "qdrant_data"
    client = QdrantClient(path=str(local_path), prefer_grpc=False)
    print(f"💾 Using local Qdrant storage at {local_path}")


def create_collection():
    # Create collection if it's missing. Safe to call multiple times.
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
    """Search for similar cases in Qdrant."""
    try:
        # Auto-create collection so the app doesn't fail with "Collection not found".
        existing = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            create_collection()

        # If collection is empty, return early to avoid noise.
        try:
            cnt = client.count(collection_name=COLLECTION_NAME).count
            if int(cnt) == 0:
                return []
        except Exception:
            # Some qdrant-client versions may not support count() for local.
            pass

        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        if hasattr(client, "search"):
            results = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=embedding,
                limit=top_k,
                with_payload=True,
            )
            points = results if isinstance(results, list) else getattr(results, "points", results)

        elif hasattr(client, "query_points"):
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=embedding,
                limit=top_k,
                with_payload=True,
            )
            points = getattr(results, "points", [])

        elif hasattr(client, "search_points"):
            results = client.search_points(
                collection_name=COLLECTION_NAME,
                query_vector=embedding,
                limit=top_k,
                with_payload=True,
            )
            points = getattr(results, "points", [])

        else:
            raise RuntimeError("No compatible Qdrant search method available")

        return [
            {
                "label": point.payload.get("label", "Unknown"),
                "image_path": point.payload.get("image_path", ""),
                "similarity_score": round(float(point.score), 3)
            }
            for point in points
        ]

    except Exception as e:
        print(f"⚠️ Qdrant search failed: {e}")
        import traceback
        traceback.print_exc()
        return []