import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, OptimizersConfigDiff, CollectionStatus
from typing import List, Optional, Dict, Any, Callable
import logging
import os
from dotenv import load_dotenv
from datetime import datetime
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai
import json
import threading

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Global cancellation tracking
cancellation_requests = {}
cancellation_lock = threading.RLock()

scraping_progress = {}
progress_lock = threading.Lock()

def update_progress(task_id: str, status: str, **kwargs):
    """Update progress for async tasks (non-thread-safe version)."""
    if task_id in scraping_progress:
        scraping_progress[task_id].update({
            "status": status,
            "last_update": datetime.now(),
            "is_completed": status in ["completed", "error", "cancelled"],
            **kwargs
        })
        
        # Set error field if provided
        if "error" in kwargs:
            scraping_progress[task_id]["error"] = kwargs["error"]
        
        # Set result field if provided
        if "result" in kwargs:
            scraping_progress[task_id]["result"] = kwargs["result"]

def request_cancellation(task_id: str):
    """Request cancellation for a specific task."""
    with cancellation_lock:
        cancellation_requests[task_id] = True
        logger.info(f"Cancellation requested for task: {task_id}")

def is_cancellation_requested(task_id: str) -> bool:
    """Check if cancellation has been requested for a task."""
    with cancellation_lock:
        return cancellation_requests.get(task_id, False)

def clear_cancellation_request(task_id: str):
    """Clear cancellation request for a task."""
    with cancellation_lock:
        cancellation_requests.pop(task_id, None)

class CancellationException(Exception):
    """Exception raised when a task is cancelled."""
    pass

def get_qdrant_client() -> QdrantClient:
    """Create and return a Qdrant client with proper configuration for local or hosted setup."""
    try:
        # Check if using hosted or local Qdrant
        use_hosted_qdrant = os.getenv("USE_HOSTED_QDRANT", "false").lower() == "true"
        
        if use_hosted_qdrant:
            # Hosted Qdrant configuration (e.g., Qdrant Cloud)
            api_key = os.getenv("HOSTED_QDRANT_API_KEY")
            url = os.getenv("HOSTED_QDRANT_URL")
            
            if not url:
                raise ValueError("HOSTED_QDRANT_URL is required when using hosted Qdrant")
            
            client_params = {
                "url": url,
                "timeout": 60,
            }
            
            # Add API key if provided
            if api_key:
                client_params["api_key"] = api_key
            
            logger.info(f"Connecting to hosted Qdrant at {url}")
            client = QdrantClient(**client_params)
            
        else:
            # Local Qdrant configuration
            host = os.getenv("LOCAL_QDRANT_HOST", "localhost")
            port = int(os.getenv("LOCAL_QDRANT_PORT", "6333"))
            
            client = QdrantClient(
                host=host,
                port=port,
                timeout=60,
                prefer_grpc=False  # Use HTTP instead of gRPC for better compatibility
            )
            logger.info(f"Connecting to local Qdrant at {host}:{port}")
        
        # Test connection
        client.get_collections()
        logger.info(f"Successfully connected to {'hosted' if use_hosted_qdrant else 'local'} Qdrant server")
        return client
        
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant client: {e}")
        raise

# Initialize Qdrant client
try:
    qdrant = get_qdrant_client()
except Exception as e:
    logger.error(f"Failed to connect to Qdrant server: {e}")
    raise

VECTOR_SIZE = 384

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def create_collection_if_not_exists(collection_name: str, task_id: Optional[str] = None) -> None:
    """Create a Qdrant collection if it doesn't exist."""
    try:
        # Check for cancellation
        if task_id and is_cancellation_requested(task_id):
            raise CancellationException(f"Task {task_id} was cancelled during collection creation")
        
        # Check if collection exists
        collections = qdrant.get_collections()
        existing_names = [col.name for col in collections.collections]
        
        if collection_name not in existing_names:
            # Create collection with proper configuration
            qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                    on_disk=True  # Store vectors on disk to save memory
                ),
                optimizers_config=OptimizersConfigDiff(
                    default_segment_number=2,
                    max_optimization_threads=4,
                    memmap_threshold=20000,
                    indexing_threshold=20000
                ),
                replication_factor=1,  # Single node setup
                write_consistency_factor=1,  # Single node setup
                init_from=None  # Don't initialize from another collection
            )
            logger.info(f"Created collection: {collection_name}")
        else:
            logger.info(f"Collection {collection_name} already exists")
            
        # Verify collection was created/accessed
        try:
            # Use get_collection instead of get_collection_info
            collection_info = qdrant.get_collection(collection_name)
            if not collection_info:
                raise Exception(f"Failed to verify collection {collection_name}")
            
            # Check if collection is ready
            if collection_info.status != CollectionStatus.GREEN:
                logger.warning(f"Collection {collection_name} status is {collection_info.status}")
                
            logger.info(f"Collection {collection_name} verified successfully")
            
        except Exception as e:
            logger.error(f"Error verifying collection: {e}")
            # Don't raise here, as the collection might still be usable
            
    except CancellationException:
        raise
    except Exception as e:
        logger.error(f"Failed to create/verify collection: {e}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def ingest_to_qdrant(
    collection_name: str, 
    texts: List[str], 
    embeddings: List[List[float]], 
    task_id: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> None:
    """Ingest text chunks and embeddings into Qdrant with cancellation support."""
    try:
        # Check for cancellation at start
        if task_id and is_cancellation_requested(task_id):
            raise CancellationException(f"Task {task_id} was cancelled before ingestion started")
        
        # Validate inputs
        if not texts or not embeddings:
            raise ValueError("Empty texts or embeddings provided")
            
        if len(texts) != len(embeddings):
            raise ValueError(f"Mismatched lengths: {len(texts)} texts vs {len(embeddings)} embeddings")
            
        # Validate embedding dimensions
        for i, embedding in enumerate(embeddings):
            if len(embedding) != VECTOR_SIZE:
                raise ValueError(f"Invalid embedding dimension at index {i}: {len(embedding)} vs {VECTOR_SIZE}")
        
        # Ensure collection exists
        create_collection_if_not_exists(collection_name, task_id)
        
        # Prepare points with metadata
        points = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            # Check for cancellation during point preparation
            if task_id and is_cancellation_requested(task_id):
                logger.info(f"Task {task_id} cancelled during point preparation at index {i}")
                raise CancellationException(f"Task {task_id} was cancelled during point preparation")
            
            if not text.strip():
                continue
                
            point = PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "text": text,
                    "metadata": {
                        "chunk_index": i,
                        "text_length": len(text),
                        "created_at": datetime.now().isoformat(),
                        "task_id": task_id  # Track which task created this data
                    }
                }
            )
            points.append(point)
        
        if not points:
            raise ValueError("No valid points to insert")
            
        # Batch process points with cancellation checks
        batch_size = 100
        total_ingested = 0
        total_batches = (len(points) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(points), batch_size):
            # Check for cancellation before each batch
            if task_id and is_cancellation_requested(task_id):
                logger.info(f"Task {task_id} cancelled after ingesting {total_ingested} points")
                raise CancellationException(f"Task {task_id} was cancelled during batch processing")
            
            batch = points[batch_idx:batch_idx + batch_size]
            current_batch_num = batch_idx // batch_size + 1
            
            try:
                qdrant.upsert(
                    collection_name=collection_name,
                    points=batch,
                    wait=True,  # Wait for operation to complete
                    ordering=None  # No specific ordering required
                )
                total_ingested += len(batch)
                logger.info(f"Successfully ingested batch {current_batch_num}/{total_batches} ({len(batch)} points)")
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(total_ingested, len(points))
                
            except Exception as e:
                logger.error(f"Failed to ingest batch {current_batch_num}: {e}")
                # Continue with next batch instead of raising
                continue
                
        if total_ingested == 0:
            raise Exception("Failed to ingest any points")
            
        logger.info(f"Successfully ingested {total_ingested}/{len(points)} points to collection {collection_name}")
        
        # Clear cancellation request if task completed successfully
        if task_id:
            clear_cancellation_request(task_id)
        
    except CancellationException:
        # Don't clear cancellation request for cancelled tasks
        logger.info(f"Ingestion cancelled for task {task_id}")
        raise
    except Exception as e:
        logger.error(f"Failed to ingest to Qdrant: {e}")
        raise

def ingest_to_qdrant_incremental(
    collection_name: str, 
    texts: List[str], 
    embeddings: List[List[float]], 
    task_id: str,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Any]:
    """
    Ingest data incrementally with detailed progress tracking and cancellation support.
    Returns information about what was actually stored.
    """
    try:
        # Check for cancellation at start
        if is_cancellation_requested(task_id):
            raise CancellationException(f"Task {task_id} was cancelled before ingestion started")
        
        # Validate inputs
        if not texts or not embeddings:
            raise ValueError("Empty texts or embeddings provided")
            
        if len(texts) != len(embeddings):
            raise ValueError(f"Mismatched lengths: {len(texts)} texts vs {len(embeddings)} embeddings")
        
        # Ensure collection exists
        create_collection_if_not_exists(collection_name, task_id)
        
        # Generate unique IDs using hash to avoid collisions
        points = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            # Check for cancellation during point preparation
            if is_cancellation_requested(task_id):
                logger.info(f"Task {task_id} cancelled during point preparation at index {i}")
                break
            
            if not text.strip():
                continue
                
            # Create unique ID using hash of content + timestamp + index
            unique_id = hashlib.md5(f"{text[:100]}_{task_id}_{i}_{datetime.now().timestamp()}".encode()).hexdigest()
            
            point = PointStruct(
                id=unique_id,  # Use hash-based unique IDs
                vector=embedding,
                payload={
                    "text": text,
                    "metadata": {
                        "chunk_index": i,
                        "text_length": len(text),
                        "created_at": datetime.now().isoformat(),
                        "task_id": task_id,
                        "collection_name": collection_name
                    }
                }
            )
            points.append(point)
        
        if not points:
            return {
                "total_points": 0,
                "ingested_points": 0,
                "status": "no_valid_points"
            }
        
        # Batch process points with cancellation checks
        batch_size = 100  # Increased batch size for better performance
        total_ingested = 0
        
        for batch_idx in range(0, len(points), batch_size):
            # Check for cancellation before each batch
            if is_cancellation_requested(task_id):
                logger.info(f"Task {task_id} cancelled after ingesting {total_ingested}/{len(points)} points")
                break
            
            batch = points[batch_idx:batch_idx + batch_size]
            
            try:
                qdrant.upsert(
                    collection_name=collection_name,
                    points=batch,
                    wait=True
                )
                total_ingested += len(batch)
                
                # Update progress
                if progress_callback:
                    progress_message = f"Stored {total_ingested}/{len(points)} chunks in Qdrant"
                    progress_callback(total_ingested, len(points), progress_message)
                
                logger.info(f"Ingested batch: {total_ingested}/{len(points)} points")
                
            except Exception as e:
                logger.error(f"Failed to ingest batch starting at {batch_idx}: {e}")
                # Continue with next batch instead of failing completely
                continue
        
        # Determine final status
        if is_cancellation_requested(task_id):
            status = "cancelled"
            clear_cancellation_request(task_id)
        elif total_ingested == len(points):
            status = "completed"
            clear_cancellation_request(task_id)
        else:
            status = "partial"
        
        result = {
            "total_points": len(points),
            "ingested_points": total_ingested,
            "status": status,
            "collection_name": collection_name
        }
        
        logger.info(f"Ingestion result for task {task_id}: {result}")
        return result
        
    except CancellationException:
        logger.info(f"Ingestion cancelled for task {task_id}")
        return {
            "total_points": len(points) if 'points' in locals() else 0,
            "ingested_points": total_ingested if 'total_ingested' in locals() else 0,
            "status": "cancelled",
            "collection_name": collection_name
        }
    except Exception as e:
        logger.error(f"Failed to ingest to Qdrant: {e}")
        raise

def query_qdrant(collection_name: str, query_vector: List[float], limit: int = 10) -> List[dict]:
    """Query top relevant chunks from Qdrant using cosine similarity."""
    try:
        hits = qdrant.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            }
            for hit in hits
        ]
    except Exception as e:
        logging.error(f"Failed to query Qdrant: {e}")
        return []
    
def enhanced_query_qdrant(collection_name: str, query_vector: List[float], keywords: List[str], limit: int = 5) -> List[dict]:
    """Enhanced query with keyword filtering and multiple search strategies."""
    try:
        # Primary vector search
        hits = qdrant.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit * 2,  # Get more results initially
            with_payload=True
        )
        
        results = []
        for hit in hits:
            score = hit.score
            payload = hit.payload
            
            # Boost score if keywords found in text
            if payload and payload.get('text'):
                text_lower = payload['text'].lower()
                keyword_matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
                if keyword_matches > 0:
                    score += (keyword_matches * 0.1)  # Boost score
            
            results.append({
                "id": hit.id,
                "score": score,
                "payload": payload,
                "keyword_matches": keyword_matches if 'keyword_matches' in locals() else 0
            })
        
        # Sort by enhanced score and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
        
    except Exception as e:
        logging.error(f"Failed to query Qdrant: {e}")
        return []