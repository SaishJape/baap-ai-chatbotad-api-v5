from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Depends, requests, status, Form
from fastapi.security import OAuth2PasswordRequestForm
from app.utils.LangChain import add_to_conversation_history, get_simple_conversation_history
from app.utils.process_files import process_pdf, process_svg, process_text_file
from app.db.models import (
    QARequest, ScrapeRequest, UserCreate, UserLogin, User, Token, 
    ChatbotCreate, ChatbotInfo, UserChatbotsResponse
)
from app.services.gemini import ask_gemini_fast, enhanced_query_with_gemini, translate_to_english
from app.services.embeddings import get_embeddings, get_question_embedding
from app.utils.common import clean_text, crawl_website, create_chunks, scrape_url, should_skip_url
from app.db.qdrant import clear_cancellation_request, ingest_to_qdrant, update_progress
from app.auth.auth import (
    get_password_hash, verify_password, create_access_token,
    get_current_active_user, ACCESS_TOKEN_EXPIRE_MINUTES
)
from app.db.mysql import get_db
import logging
from typing import Dict, List, Optional
import asyncio
from datetime import datetime, timedelta
import hashlib
import uuid
import traceback
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from app.db.qdrant import (
    ingest_to_qdrant_incremental, 
    request_cancellation, 
    is_cancellation_requested,
    CancellationException
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Thread-safe storage for scraping progress with locks
scraping_progress: Dict[str, dict] = {}
progress_lock = threading.RLock()  # Re-entrant lock for nested operations

# Thread pool for concurrent scraping tasks
scraping_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="scraper")

# Cleanup interval for old tasks (in seconds)
CLEANUP_INTERVAL = 3600  # 1 hour
MAX_TASK_AGE = 7200  # 2 hours

# Allowed file types
ALLOWED_EXTENSIONS = {
    'pdf': 'application/pdf',
    'svg': 'image/svg+xml',
    'txt': 'text/plain',
    'doc': 'application/msword',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
}

def cleanup_old_tasks():
    """Remove old completed or errored tasks from memory."""
    current_time = datetime.now()
    with progress_lock:
        tasks_to_remove = []
        for task_id, task_data in scraping_progress.items():
            task_age = (current_time - task_data["start_time"]).total_seconds()
            if task_age > MAX_TASK_AGE and task_data.get("is_completed", False):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del scraping_progress[task_id]
            logger.info(f"Cleaned up old task: {task_id}")

def get_progress_safely(task_id: str) -> dict:
    """Thread-safe way to get progress data."""
    with progress_lock:
        return scraping_progress.get(task_id, {}).copy()


def update_progress_safely(task_id: str, status: str, **kwargs):
    """Thread-safe progress update function."""
    with progress_lock:
        if task_id in scraping_progress:
            scraping_progress[task_id].update({
                "status": status,
                "last_update": datetime.now(),
                **kwargs
            })
            
            # Set completion flag for final states
            if status in ["completed", "cancelled", "error", "partial_completed"]:
                scraping_progress[task_id]["is_completed"] = True

# Background cleanup task
async def periodic_cleanup():
    """Periodically clean up old tasks."""
    while True:
        try:
            cleanup_old_tasks()
            await asyncio.sleep(CLEANUP_INTERVAL)
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")
            await asyncio.sleep(60)  # Retry after 1 minute on error

# Start cleanup task when the module loads
cleanup_task = None

@router.on_event("startup")
async def startup_event():
    global cleanup_task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    logger.info("Started periodic cleanup task for scraping progress")

@router.on_event("shutdown")
async def shutdown_event():
    global cleanup_task
    if cleanup_task:
        cleanup_task.cancel()
    scraping_executor.shutdown(wait=True)
    logger.info("Shutdown scraping executor and cleanup task")

@router.post("/signup", response_model=User)
async def signup(user: UserCreate, db = Depends(get_db)):
    """Create a new user account."""
    try:
        logger.info(f"Signup attempt for email: {user.email}")
        
        # Check if user already exists
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s", (user.email,))
        existing_user = cursor.fetchone()
        cursor.close()
        
        if existing_user:
            logger.warning(f"Signup failed: Email already registered: {user.email}")
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create new user
        user_id = str(uuid.uuid4())
        hashed_password = get_password_hash(user.password)
        
        # Insert new user
        cursor = db.cursor()
        cursor.execute("""
            INSERT INTO users (id, email, username, password_hash)
            VALUES (%s, %s, %s, %s)
        """, (
            user_id,
            user.email,
            user.username,
            hashed_password
        ))
        db.commit()
        cursor.close()
        
        logger.info(f"User created successfully: {user.email}")
        return {
            'id': user_id,
            'email': user.email,
            'username': user.username,
            'created_at': datetime.now(),
            'is_active': True
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in signup: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="An error occurred during signup. Please try again."
        )

@router.post("/login", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db = Depends(get_db)
):
    """Login endpoint for OAuth2 password flow"""
    try:
        logger.info(f"Login attempt for username: {form_data.username}")
        
        # Find user
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s", (form_data.username,))
        user = cursor.fetchone()
        cursor.close()
        
        if not user:
            logger.warning(f"Login failed: User not found with email {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Verify password
        if not verify_password(form_data.password, user['password_hash']):
            logger.warning(f"Login failed: Invalid password for user {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user['id']}, expires_delta=access_token_expires
        )
        
        logger.info(f"Login successful for user {form_data.username}")
        return {"access_token": access_token, "token_type": "bearer"}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in login: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during login. Please try again."
        )

@router.post("/login/json", response_model=Token)
async def login_json(user_data: UserLogin, db = Depends(get_db)):
    """Login endpoint for JSON data"""
    try:
        logger.info(f"Login attempt for email: {user_data.email}")
        
        # Find user
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s", (user_data.email,))
        user = cursor.fetchone()
        cursor.close()
        
        if not user:
            logger.warning(f"Login failed: User not found with email {user_data.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Verify password
        if not verify_password(user_data.password, user['password_hash']):
            logger.warning(f"Login failed: Invalid password for user {user_data.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user['id']}, expires_delta=access_token_expires
        )
        
        logger.info(f"Login successful for user {user_data.email}")
        return {"access_token": access_token, "token_type": "bearer"}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in login: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during login. Please try again."
        )

# NEW ENDPOINTS FOR MULTI-CHATBOT SUPPORT

@router.get("/user/chatbots", response_model=UserChatbotsResponse)
async def get_user_chatbots(
    current_user = Depends(get_current_active_user),
    db = Depends(get_db)
):
    """Get all chatbots for the current user."""
    try:
        cursor = db.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM chatbots 
            WHERE user_id = %s AND is_active = TRUE 
            ORDER BY created_at DESC
        """, (current_user['id'],))
        
        chatbots = cursor.fetchall()
        cursor.close()
        
        return {
            "chatbots": chatbots,
            "total_count": len(chatbots)
        }
        
    except Exception as e:
        logger.error(f"Error fetching user chatbots: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching chatbots")

@router.post("/chatbots", response_model=ChatbotInfo)
async def create_chatbot(
    chatbot_data: ChatbotCreate,
    current_user = Depends(get_current_active_user),
    db = Depends(get_db)
):
    """Create a new chatbot record."""
    try:
        chatbot_id = str(uuid.uuid4())
        
        cursor = db.cursor()
        cursor.execute("""
            INSERT INTO chatbots (id, user_id, name, description, collection_name, source_url)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            chatbot_id,
            current_user['id'],
            chatbot_data.name,
            chatbot_data.description,
            chatbot_data.collection_name,
            chatbot_data.source_url
        ))
        db.commit()
        cursor.close()
        
        # Return the created chatbot
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM chatbots WHERE id = %s", (chatbot_id,))
        chatbot = cursor.fetchone()
        cursor.close()
        
        return chatbot
        
    except Exception as e:
        logger.error(f"Error creating chatbot: {str(e)}")
        raise HTTPException(status_code=500, detail="Error creating chatbot")

@router.delete("/chatbots/{chatbot_id}")
async def delete_chatbot(
    chatbot_id: str,
    current_user = Depends(get_current_active_user),
    db = Depends(get_db)
):
    """Delete a chatbot (soft delete by setting is_active to False)."""
    try:
        logger.info(f"Delete request for chatbot {chatbot_id} by user {current_user['id']}")
        
        # First, check if the chatbot exists and belongs to the current user
        cursor = db.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM chatbots 
            WHERE id = %s AND user_id = %s AND is_active = TRUE
        """, (chatbot_id, current_user['id']))
        
        chatbot = cursor.fetchone()
        cursor.close()
        
        if not chatbot:
            logger.warning(f"Chatbot not found or access denied: {chatbot_id}")
            raise HTTPException(
                status_code=404, 
                detail="Chatbot not found or you don't have permission to delete it"
            )
        
        # Perform soft delete by setting is_active to FALSE
        cursor = db.cursor()
        cursor.execute("""
            UPDATE chatbots 
            SET is_active = FALSE, updated_at = %s
            WHERE id = %s AND user_id = %s
        """, (datetime.now(), chatbot_id, current_user['id']))
        
        db.commit()
        
        if cursor.rowcount == 0:
            cursor.close()
            logger.error(f"Failed to delete chatbot: {chatbot_id}")
            raise HTTPException(status_code=500, detail="Failed to delete chatbot")
        
        cursor.close()
        
        logger.info(f"Chatbot {chatbot_id} successfully deleted by user {current_user['id']}")
        
        return {
            "status": "success",
            "message": "Chatbot deleted successfully",
            "chatbot_id": chatbot_id,
            "chatbot_name": chatbot.get('name', 'Unknown')
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error deleting chatbot {chatbot_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail="An error occurred while deleting the chatbot"
        )

@router.post("/upload-and-process")
async def upload_and_process(
    file: UploadFile = File(...),
    collection_name: Optional[str] = Form(None),  # Accept as form data
    current_user = Depends(get_current_active_user),
    background_tasks: BackgroundTasks = None,
    db = Depends(get_db)
):
    """Upload and process files (PDF, SVG, etc.) and store in specified collection."""
    try:
        # If no collection_name provided, use user's default collection
        if not collection_name:
            collection_name = f"{current_user['id']}_default"
        
        logger.info(f"Starting file upload process for file: {file.filename} to collection: {collection_name}")
        
        # Validate file type
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            logger.warning(f"Invalid file type: {file_extension}")
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS.keys())}"
            )

        # Read file content
        try:
            file_content = await file.read()
            if not file_content:
                raise HTTPException(status_code=400, detail="Empty file received")
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
        
        # Process file based on type
        text_content = ""
        try:
            if file_extension == 'pdf':
                text_content = process_pdf(file_content)
            elif file_extension == 'svg':
                text_content = process_svg(file_content)
            else:
                text_content = process_text_file(file_content)
        except Exception as e:
            logger.error(f"Error processing file content: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing file content: {str(e)}")

        if not text_content.strip():
            logger.warning("No text content extracted from file")
            raise HTTPException(status_code=400, detail="No text content could be extracted from the file")

        # Create chunks from the text
        try:
            chunks = create_chunks(text_content, chunk_size=1000, overlap=200)
            logger.info(f"Created {len(chunks)} chunks from text")
            
            # Validate chunks
            valid_chunks = [chunk for chunk in chunks if chunk.strip()]
            if len(valid_chunks) != len(chunks):
                logger.warning(f"Filtered out {len(chunks) - len(valid_chunks)} empty chunks")
                chunks = valid_chunks
                
            if not chunks:
                raise HTTPException(status_code=400, detail="No valid text chunks could be created from the file")
                
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error creating text chunks: {str(e)}")

        # Generate embeddings
        try:
            embeddings = get_embeddings(chunks)
            if not embeddings or len(embeddings) != len(chunks):
                raise Exception("Embedding generation failed or produced mismatched results")
            logger.info(f"Generated {len(embeddings)} embeddings")
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

        # Ingest to Qdrant
        try:
            ingest_to_qdrant(collection_name, chunks, embeddings)
            logger.info(f"Successfully ingested {len(chunks)} chunks to collection {collection_name}")
        except Exception as e:
            logger.error(f"Error ingesting to Qdrant: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error storing data: {str(e)}")

        return {
            "status": "success",
            "message": "File processed and stored successfully",
            "collection_name": collection_name,
            "chunks_created": len(chunks),
            "file_name": file.filename
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in upload_and_process: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/scrape-and-ingest")
async def scrape_and_ingest(
    req: ScrapeRequest,
    current_user = Depends(get_current_active_user),
    background_tasks: BackgroundTasks = None,
    db = Depends(get_db)
):
    """Scrape a website and ingest the content into specified collection."""
    try:
        # Use collection_name from request (now required from frontend)
        collection_name = req.collection_name
        
        logger.info(f"Starting scrape and ingest for URL: {req.url} to collection: {collection_name}")
        
        # Generate a unique ID for this scraping task
        task_id = hashlib.md5(f"{req.url}_{collection_name}_{datetime.now().timestamp()}".encode()).hexdigest()
        
        # Initialize progress tracking with thread safety
        with progress_lock:
            scraping_progress[task_id] = {
                "status": "queued",
                "start_time": datetime.now(),
                "last_update": datetime.now(),
                "pages_scraped": 0,
                "chunks_created": 0,
                "error": None,
                "is_completed": False,
                "collection_name": collection_name,
                "url": req.url,
                "user_id": current_user['id']  # Track which user started this task
            }
        
        # Submit task to thread pool executor
        future = scraping_executor.submit(process_scraping_sync, req.url, task_id, collection_name)
        
        # Don't wait for the result, just return the task_id immediately
        logger.info(f"Scraping task {task_id} submitted to thread pool")
        
        return {
            "task_id": task_id, 
            "status": "queued",
            "collection_name": collection_name,
            "message": "Scraping task has been queued and will start shortly"
        }
        
    except Exception as e:
        logger.error(f"Error starting scrape process: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scraping-progress/{task_id}")
async def get_scraping_progress(
    task_id: str,
    current_user = Depends(get_current_active_user)
):
    """Get the current progress of a scraping task."""
    progress_data = get_progress_safely(task_id)
    
    if not progress_data:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Check if the current user owns this task (optional security check)
    if progress_data.get("user_id") != current_user['id']:
        raise HTTPException(status_code=403, detail="Access denied to this task")
    
    # Check if we should return cached response
    current_time = datetime.now()
    last_update = progress_data["last_update"]
    time_diff = (current_time - last_update).total_seconds()
    
    # If the task is completed or there's an error, return immediately
    if progress_data.get("is_completed", False) or progress_data.get("error"):
        return progress_data
    
    # If less than 2 seconds have passed since last update, return cached response
    if time_diff < 2:
        return progress_data
    
    # Update last access time (but don't change the actual last_update from the worker)
    return progress_data

@router.get("/scraping-tasks")
async def get_user_scraping_tasks(
    current_user = Depends(get_current_active_user),
    include_completed: bool = False
):
    """Get all scraping tasks for the current user."""
    user_tasks = {}
    
    with progress_lock:
        for task_id, task_data in scraping_progress.items():
            if task_data.get("user_id") == current_user['id']:
                if include_completed or not task_data.get("is_completed", False):
                    user_tasks[task_id] = task_data.copy()
    
    return {
        "tasks": user_tasks,
        "total_count": len(user_tasks)
    }

from sentence_transformers import SentenceTransformer

def process_scraping_sync(url: str, task_id: str, collection_name: str):
    """Process scraping with incremental storage - store data as it's crawled."""
    try:
        update_progress_safely(task_id, "starting", message="Initializing scraping process...")
        
        # Initialize counters
        total_pages_crawled = 0
        total_chunks_stored = 0
        
        # Initialize embeddings model
        embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Start crawling with incremental processing
        visited = set()
        urls_to_visit = [url]
        start_domain = urlparse(url).netloc
        
        while urls_to_visit and not is_cancellation_requested(task_id):
            current_url = urls_to_visit.pop(0)
            
            if current_url in visited:
                continue
                
            try:
                # Update progress
                update_progress_safely(task_id, "crawling", 
                                     message=f"Crawling page {total_pages_crawled + 1}: {current_url}",
                                     pages_scraped=total_pages_crawled)
                
                # Scrape current page
                html_content = scrape_url(current_url)
                if not html_content:
                    continue
                
                # Check if it's HTML content
                try:
                    response_check = requests.head(current_url, timeout=5)
                    content_type = response_check.headers.get("Content-Type", "")
                    if "text/html" not in content_type:
                        continue
                except:
                    pass
                
                visited.add(current_url)
                total_pages_crawled += 1
                
                # IMMEDIATELY process this page's content
                clean_text_content = clean_text(html_content)
                if clean_text_content:
                    # Create chunks from this page
                    page_chunks = create_chunks(clean_text_content, chunk_size=1000, overlap=200)
                    
                    if page_chunks:
                        # Check for cancellation before processing chunks
                        if is_cancellation_requested(task_id):
                            logger.info(f"Task {task_id} cancelled during chunk processing for page {current_url}")
                            break
                        
                        # Generate embeddings for these chunks
                        try:
                            embeddings = embeddings_model.encode(page_chunks).tolist()
                            
                            # Store chunks immediately
                            result = ingest_to_qdrant_incremental(
                                collection_name=collection_name,
                                texts=page_chunks,
                                embeddings=embeddings,
                                task_id=task_id,
                                progress_callback=lambda stored, total, msg: update_progress_safely(
                                    task_id, "processing", 
                                    message=f"Stored {stored}/{total} chunks from page {total_pages_crawled}",
                                    pages_scraped=total_pages_crawled,
                                    chunks_created=total_chunks_stored + stored
                                )
                            )
                            
                            # Update chunk count
                            chunks_from_this_page = result.get("ingested_points", 0)
                            total_chunks_stored += chunks_from_this_page
                            
                            logger.info(f"Page {total_pages_crawled}: Stored {chunks_from_this_page} chunks. Total: {total_chunks_stored}")
                            
                        except Exception as e:
                            logger.error(f"Error processing chunks for {current_url}: {e}")
                            # Continue with next page even if this one fails
                
                # Extract links for further crawling (only if not cancelled)
                if not is_cancellation_requested(task_id):
                    soup = BeautifulSoup(html_content, "html.parser")
                    for link in soup.find_all("a", href=True):
                        full_url = urljoin(current_url, link["href"])
                        
                        # Only crawl links from the same domain
                        if (urlparse(full_url).netloc == start_domain
                            and full_url not in visited
                            and full_url not in urls_to_visit
                            and not should_skip_url(full_url)):
                            urls_to_visit.append(full_url)
                
                # Update progress after each page
                update_progress_safely(task_id, "crawling", 
                                     message=f"Crawled {total_pages_crawled} pages, stored {total_chunks_stored} chunks",
                                     pages_scraped=total_pages_crawled,
                                     chunks_created=total_chunks_stored)
                
                # Small delay to make cancellation more responsive
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing {current_url}: {e}")
                continue
        
        # Final status update
        if is_cancellation_requested(task_id):
            final_status = "cancelled"
            final_message = f"Scraping cancelled. Processed {total_pages_crawled} pages, stored {total_chunks_stored} chunks"
        else:
            final_status = "completed"
            final_message = f"Scraping completed. Processed {total_pages_crawled} pages, stored {total_chunks_stored} chunks"
        
        update_progress_safely(task_id, final_status,
                             message=final_message,
                             pages_scraped=total_pages_crawled,
                             chunks_created=total_chunks_stored,
                             is_completed=True,
                             result={
                                 "total_pages": total_pages_crawled,
                                 "total_chunks": total_chunks_stored,
                                 "collection_name": collection_name,
                                 "status": final_status
                             })
        
        logger.info(f"Task {task_id} {final_status}: {total_pages_crawled} pages, {total_chunks_stored} chunks")
        
    except CancellationException:
        logger.info(f"Task {task_id} was cancelled during processing")
        update_progress_safely(task_id, "cancelled",
                             message="Task cancelled by user",
                             pages_scraped=total_pages_crawled if 'total_pages_crawled' in locals() else 0,
                             chunks_created=total_chunks_stored if 'total_chunks_stored' in locals() else 0,
                             is_completed=True)
    except Exception as e:
        logger.error(f"Error in scraping task {task_id}: {e}")
        update_progress_safely(task_id, "error",
                             error=str(e),
                             pages_scraped=total_pages_crawled if 'total_pages_crawled' in locals() else 0,
                             chunks_created=total_chunks_stored if 'total_chunks_stored' in locals() else 0,
                             is_completed=True)


@router.post("/ask-question")
async def ask_question(req: QARequest):
    try:
        # Input validation
        if not req.question or req.question.strip() == "":
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if not req.collection_name or req.collection_name.strip() == "":
            raise HTTPException(status_code=400, detail="Collection name cannot be empty")

        logger.info(f"Processing question: '{req.question}' for collection: '{req.collection_name}'")
        
        # Get conversation history (super fast)
        conversation_history = get_simple_conversation_history(req.collection_name)
        logger.debug(f"Retrieved {len(conversation_history)} messages from history")

        # Translate query if needed
        try:
            translated_query = translate_to_english(req.question)
            logger.debug(f"Translated query: {translated_query}")
        except Exception as e:
            logger.warning(f"Translation failed, using original query: {e}")
            translated_query = req.question

        # Step 1: Get embedding
        try:
            question_embedding = get_question_embedding(translated_query)
            logger.debug(f"Generated embedding with {len(question_embedding)} dimensions")
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise HTTPException(status_code=500, detail="Failed to process question")

        # Step 2: Enhanced query to get search results and context
        try:
            enhanced_results = enhanced_query_with_gemini(
                collection_name=req.collection_name,
                user_query=translated_query,
                query_vector=question_embedding,
                limit=10
            )
            logger.info(f"Enhanced query returned {enhanced_results.get('total_results', 0)} results")
        except Exception as e:
            logger.error(f"Enhanced query failed: {e}")
            enhanced_results = {
                "error": str(e),
                "context_text": "Failed to retrieve relevant context.",
                "processed_query": {},
                "search_results": []
            }

        # Step 3: Ask Gemini with simple conversation history
        try:
            final_response = ask_gemini_fast(
                context=enhanced_results.get("context_text", ""),
                question=req.question,
                query_analysis=enhanced_results.get("processed_query", {}),
                enhanced_results=enhanced_results,
                conversation_history=conversation_history
            )
            logger.debug("Successfully generated Gemini response")
        except Exception as e:
            logger.error(f"Gemini response generation failed: {e}")
            final_response = {
                "response": "I apologize, but I'm unable to process your request at the moment. Please try again later.",
                "buttons": False,
                "button_type": None,
                "button_data": None
            }

        # Save to conversation history (super fast)
        try:
            add_to_conversation_history(req.collection_name, "user", req.question)
            add_to_conversation_history(req.collection_name, "assistant", final_response["response"])
            logger.debug("Saved conversation to simple memory")
        except Exception as e:
            logger.warning(f"Failed to save to memory: {e}")

        # Add debug info to response
        final_response["conversation_id"] = req.collection_name
        final_response["debug_info"] = {
            "total_search_results": enhanced_results.get("total_results", 0),
            "context_chunks": enhanced_results.get("context_chunks_count", 0),
            "translated_query": translated_query,
            "has_context": bool(enhanced_results.get("context_text", "").strip()),
            "memory_enabled": True,
            "memory_type": "simple_in_memory",
            "conversation_length": len(conversation_history)
        }
        
        logger.info(f"Successfully processed question with {final_response['debug_info']['total_search_results']} search results")
        return final_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in ask_question: {e}")
        return {
            "response": "Something went wrong while answering your question. Please try again later.",
            "buttons": False,
            "button_type": None,
            "button_data": None,
            "conversation_id": None,
            "debug_info": {
                "error": str(e),
                "total_search_results": 0,
                "context_chunks": 0,
                "has_context": False,
                "memory_enabled": False,
                "memory_type": "simple_in_memory"
            }
        }
import asyncio
from asyncio import Task
from typing import Dict, Optional
import threading
import time

# Add these global variables to track running tasks and cancellation flags
running_tasks: Dict[str, Task] = {}
cancellation_flags: Dict[str, threading.Event] = {}

@router.post("/stop-scraping/{task_id}")
async def stop_scraping(
    task_id: str,
    current_user = Depends(get_current_active_user),
    db = Depends(get_db)
):
    """Stop a running scraping task and deactivate associated chatbot."""
    try:
        logger.info(f"Stop scraping request for task {task_id} by user {current_user['id']}")
        
        # Check if task exists in progress tracking
        with progress_lock:
            if task_id not in scraping_progress:
                logger.warning(f"Task {task_id} not found in progress tracking")
                raise HTTPException(
                    status_code=404, 
                    detail=f"Scraping task {task_id} not found"
                )
            
            # Check if user has permission to stop this task
            task_info = scraping_progress[task_id]
            if task_info.get("user_id") != current_user['id']:
                logger.warning(f"User {current_user['id']} attempted to stop task {task_id} owned by user {task_info.get('user_id')}")
                raise HTTPException(
                    status_code=403, 
                    detail="You don't have permission to stop this task"
                )
            
            # Check if task is already completed or cancelled
            current_status = task_info.get("status", "unknown")
            if current_status in ["completed", "cancelled", "error"]:
                logger.info(f"Task {task_id} is already in final state: {current_status}")
                return {
                    "message": f"Task {task_id} is already {current_status}",
                    "status": current_status,
                    "task_id": task_id
                }
            
            # Get collection name from task info for chatbot lookup
            collection_name = task_info.get("collection_name")
        
        # Request cancellation in the Qdrant module
        request_cancellation(task_id)
        logger.info(f"Cancellation requested for task {task_id}")
        
        # Complete the cancellation process directly
        cancellation_time = datetime.now()
        
        # Update progress to cancelled state immediately
        update_progress_safely(task_id, "cancelled", 
                         error="Task cancelled by user",
                         cancellation_time=cancellation_time,
                         is_completed=True)
        
        # Clean up running task if it exists
        if task_id in running_tasks:
            try:
                running_tasks[task_id].cancel()
                del running_tasks[task_id]
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up running task {task_id}: {cleanup_error}")
        
        # Clean up cancellation flag if it exists
        if task_id in cancellation_flags:
            try:
                cancellation_flags[task_id].set()
                del cancellation_flags[task_id]
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up cancellation flag {task_id}: {cleanup_error}")
        
        # Update chatbot is_active to false if collection_name exists
        chatbot_updated = False
        if collection_name:
            try:
                cursor = db.cursor()
                cursor.execute("""
                    UPDATE chatbots 
                    SET is_active = FALSE, updated_at = %s
                    WHERE collection_name = %s AND user_id = %s AND is_active = TRUE
                """, (cancellation_time, collection_name, current_user['id']))
                
                db.commit()
                
                if cursor.rowcount > 0:
                    chatbot_updated = True
                    logger.info(f"Chatbot with collection '{collection_name}' deactivated for user {current_user['id']}")
                else:
                    logger.info(f"No active chatbot found with collection '{collection_name}' for user {current_user['id']}")
                
                cursor.close()
                
            except Exception as db_error:
                logger.error(f"Error updating chatbot status for collection '{collection_name}': {db_error}")
                # Don't raise the exception here as the main scraping cancellation was successful
        
        logger.info(f"Scraping task {task_id} has been successfully cancelled and cleaned up")
        
        response = {
            "message": f"Scraping task {task_id} has been successfully cancelled",
            "status": "cancelled",
            "task_id": task_id,
            "timestamp": cancellation_time.isoformat()
        }
        
        # Add chatbot update status to response
        if collection_name:
            response["chatbot_deactivated"] = chatbot_updated
            response["collection_name"] = collection_name
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error stopping scraping task {task_id}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to stop scraping task: {str(e)}"
        )

@router.post("/stop-and-store-scraping/{task_id}")
async def stop_and_store_scraping(
    task_id: str,
    current_user = Depends(get_current_active_user),
    db = Depends(get_db)
):
    """Stop a running scraping task and store only the pages crawled so far."""
    try:
        logger.info(f"Stop and store scraping request for task {task_id} by user {current_user['id']}")
        
        # Check if task exists in progress tracking
        with progress_lock:
            if task_id not in scraping_progress:
                logger.warning(f"Task {task_id} not found in progress tracking")
                raise HTTPException(
                    status_code=404, 
                    detail=f"Scraping task {task_id} not found"
                )
            
            # Check if user has permission to stop this task
            task_info = scraping_progress[task_id]
            if task_info.get("user_id") != current_user['id']:
                logger.warning(f"User {current_user['id']} attempted to stop task {task_id} owned by user {task_info.get('user_id')}")
                raise HTTPException(
                    status_code=403, 
                    detail="You don't have permission to stop this task"
                )
            
            # Check if task is already completed or cancelled
            current_status = task_info.get("status", "unknown")
            if current_status in ["completed", "cancelled", "error"]:
                logger.info(f"Task {task_id} is already in final state: {current_status}")
                return {
                    "message": f"Task {task_id} is already {current_status}",
                    "status": current_status,
                    "task_id": task_id,
                    "pages_stored": task_info.get("pages_scraped", 0),
                    "chunks_stored": task_info.get("chunks_created", 0)
                }
            
            # Get collection name and current progress
            collection_name = task_info.get("collection_name")
            pages_scraped = task_info.get("pages_scraped", 0)
            chunks_created = task_info.get("chunks_created", 0)
        
        # Request cancellation but mark for partial storage
        request_cancellation(task_id)
        logger.info(f"Cancellation with partial storage requested for task {task_id}")
        
        # Update status to indicate we're stopping and storing partial data
        update_progress_safely(task_id, "stopping_and_storing", 
                             message="Stopping scraping and storing crawled pages...")
        
        # Wait a brief moment for the scraping thread to process the cancellation
        # and complete any partial ingestion
        await asyncio.sleep(2)
        
        # Check final status after cancellation processing
        with progress_lock:
            final_task_info = scraping_progress.get(task_id, {})
            final_pages = final_task_info.get("pages_scraped", pages_scraped)
            final_chunks = final_task_info.get("chunks_created", chunks_created)
            ingestion_result = final_task_info.get("result", {})
        
        # Complete the cancellation process
        cancellation_time = datetime.now()
        
        # Update progress to partial completion state
        update_progress_safely(task_id, "partial_completed", 
                             message=f"Scraping stopped. Stored {final_pages} pages with {final_chunks} chunks",
                             cancellation_time=cancellation_time,
                             is_completed=True)
        
        # Clean up running task if it exists
        if task_id in running_tasks:
            try:
                running_tasks[task_id].cancel()
                del running_tasks[task_id]
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up running task {task_id}: {cleanup_error}")
        
        # Clean up cancellation flag if it exists
        if task_id in cancellation_flags:
            try:
                cancellation_flags[task_id].set()
                del cancellation_flags[task_id]
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up cancellation flag {task_id}: {cleanup_error}")
        
        # Update chatbot to active if we have stored data and collection exists
        chatbot_updated = False
        if collection_name and final_chunks > 0:
            try:
                cursor = db.cursor()
                
                # Check if chatbot already exists for this collection
                cursor.execute("""
                    SELECT id FROM chatbots 
                    WHERE collection_name = %s AND user_id = %s
                """, (collection_name, current_user['id']))
                
                existing_chatbot = cursor.fetchone()
                
                if existing_chatbot:
                    # Update existing chatbot to active
                    cursor.execute("""
                        UPDATE chatbots 
                        SET is_active = TRUE, updated_at = %s
                        WHERE collection_name = %s AND user_id = %s
                    """, (cancellation_time, collection_name, current_user['id']))
                    chatbot_updated = True
                    logger.info(f"Chatbot with collection '{collection_name}' activated for partial data")
                else:
                    # Create new chatbot entry for the partial data
                    cursor.execute("""
                        INSERT INTO chatbots (user_id, name, collection_name, is_active, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        current_user['id'], 
                        f"Partial Scrape - {collection_name}", 
                        collection_name, 
                        True, 
                        cancellation_time, 
                        cancellation_time
                    ))
                    chatbot_updated = True
                    logger.info(f"New chatbot created for partial collection '{collection_name}'")
                
                db.commit()
                cursor.close()
                
            except Exception as db_error:
                logger.error(f"Error updating chatbot status for collection '{collection_name}': {db_error}")
                # Don't raise the exception here as the main operation was successful
        
        logger.info(f"Scraping task {task_id} stopped and {final_pages} pages stored successfully")
        
        response = {
            "message": f"Scraping stopped and partial data stored successfully",
            "status": "partial_completed",
            "task_id": task_id,
            "pages_stored": final_pages,
            "chunks_stored": final_chunks,
            "collection_name": collection_name,
            "timestamp": cancellation_time.isoformat()
        }
        
        # Add ingestion details if available
        if ingestion_result:
            response["ingestion_details"] = {
                "total_points": ingestion_result.get("total_points", 0),
                "ingested_points": ingestion_result.get("ingested_points", 0),
                "ingestion_status": ingestion_result.get("status", "unknown")
            }
        
        # Add chatbot status to response
        if collection_name:
            response["chatbot_available"] = chatbot_updated and final_chunks > 0
            if not chatbot_updated and final_chunks > 0:
                response["chatbot_note"] = "Data stored but chatbot activation failed"
            elif final_chunks == 0:
                response["chatbot_note"] = "No data stored, chatbot not available"
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error stopping and storing scraping task {task_id}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to stop and store scraping task: {str(e)}"
        )

@router.delete("/scraping-tasks/{task_id}")
async def delete_scraping_task(
    task_id: str,
    current_user = Depends(get_current_active_user)
):
    """Delete a scraping task from progress tracking (for completed/cancelled tasks)."""
    try:
        logger.info(f"Delete task {task_id} request by user {current_user['id']}")
        
        with progress_lock:
            if task_id not in scraping_progress:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Task {task_id} not found"
                )
            
            task_info = scraping_progress[task_id]
            
            # Check user permission
            if task_info.get("user_id") != current_user['id']:
                raise HTTPException(
                    status_code=403, 
                    detail="You don't have permission to delete this task"
                )
            
            # Only allow deletion of completed/cancelled/error tasks
            status = task_info.get("status", "unknown")
            if status not in ["completed", "cancelled", "error", "partial_completed"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot delete active task. Current status: {status}. Stop the task first."
                )
            
            # Remove from progress tracking
            del scraping_progress[task_id]
            
        # Also clear any remaining cancellation requests
        clear_cancellation_request(task_id)
        
        logger.info(f"Task {task_id} deleted successfully")
        
        return {
            "message": f"Task {task_id} deleted successfully",
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting task {task_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to delete task: {str(e)}"
        )

@router.get("/scraping-tasks")
async def get_all_scraping_tasks(
    current_user = Depends(get_current_active_user)
):
    """Get all scraping tasks with their current status."""
    try:
        # Filter tasks and add additional info
        tasks_info = []
        current_time = datetime.now()
        
        for task_id, task_data in scraping_progress.items():
            task_info = {
                "task_id": task_id,
                "status": task_data["status"],
                "collection_name": task_data.get("collection_name", "unknown"),
                "url": task_data.get("url", "unknown"),
                "start_time": task_data["start_time"].isoformat(),
                "last_update": task_data["last_update"].isoformat(),
                "pages_scraped": task_data.get("pages_scraped", 0),
                "chunks_created": task_data.get("chunks_created", 0),
                "is_completed": task_data["is_completed"],
                "error": task_data.get("error"),
                "duration_seconds": (current_time - task_data["start_time"]).total_seconds(),
                "is_running": task_id in running_tasks and not running_tasks[task_id].done(),
                "can_be_cancelled": (task_id in running_tasks and not running_tasks[task_id].done()) or task_data["status"] in ["crawling", "processing", "generating_embeddings", "storing"]
            }
            tasks_info.append(task_info)
        
        # Sort by start time (newest first)
        tasks_info.sort(key=lambda x: x["start_time"], reverse=True)
        
        return {
            "tasks": tasks_info,
            "total_count": len(tasks_info),
            "active_count": len([t for t in tasks_info if not t["is_completed"]]),
            "completed_count": len([t for t in tasks_info if t["is_completed"] and not t["error"]]),
            "error_count": len([t for t in tasks_info if t["error"]]),
            "running_count": len([t for t in tasks_info if t["is_running"]])
        }
        
    except Exception as e:
        logger.error(f"Error fetching scraping tasks: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error fetching scraping tasks"
        )

# Modified scrape_and_ingest function to track tasks
@router.post("/scrape-and-ingest")
async def scrape_and_ingest(
    req: ScrapeRequest,
    current_user = Depends(get_current_active_user),
    background_tasks: BackgroundTasks = None,
    db = Depends(get_db)
):
    """Scrape a website and ingest the content into specified collection."""
    try:
        collection_name = req.collection_name
        
        logger.info(f"Starting scrape and ingest for URL: {req.url} to collection: {collection_name}")
        
        task_id = hashlib.md5(f"{req.url}_{collection_name}_{datetime.now().timestamp()}".encode()).hexdigest()
        
        # Initialize progress tracking
        scraping_progress[task_id] = {
            "status": "crawling",
            "start_time": datetime.now(),
            "last_update": datetime.now(),
            "pages_scraped": 0,
            "chunks_created": 0,
            "error": None,
            "is_completed": False,
            "collection_name": collection_name,
            "url": req.url,
            "user_id": current_user['id']  # Add user tracking
        }
        
        # Create cancellation flag for this task
        cancellation_flags[task_id] = threading.Event()
        
        # Create and track the background task
        task = asyncio.create_task(process_scraping_async(req.url, task_id, collection_name))
        running_tasks[task_id] = task
        
        # Add cleanup callback
        task.add_done_callback(lambda t: cleanup_task(task_id))
        
        return {
            "task_id": task_id, 
            "status": "started",
            "collection_name": collection_name
        }
        
    except Exception as e:
        logger.error(f"Error starting scrape process: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_scraping_async(url: str, task_id: str, collection_name: str):
    """Improved async version of process_scraping with better chunk handling."""
    try:
        update_progress(task_id, "crawling")
        
        # Check for cancellation
        if task_id in cancellation_flags and cancellation_flags[task_id].is_set():
            update_progress(task_id, "cancelled", error="Task was cancelled")
            return
        
        # Crawl the website with cancellation support
        pages = await asyncio.get_event_loop().run_in_executor(
            None, crawl_website_with_cancellation, str(url), task_id, None
        )
        
        # Check if crawling was cancelled
        if task_id in cancellation_flags and cancellation_flags[task_id].is_set():
            update_progress(task_id, "cancelled", error="Task was cancelled during crawling")
            return
        
        if not pages:
            update_progress(task_id, "error", error="No pages could be scraped from the provided URL")
            return
        
        update_progress(task_id, "crawling", pages_scraped=len(pages))
        update_progress(task_id, "processing")
        
        all_chunks = []
        
        # Process each page with improved chunking
        for i, (url_page, html) in enumerate(pages.items()):
            # Check for cancellation during processing
            if task_id in cancellation_flags and cancellation_flags[task_id].is_set():
                update_progress(task_id, "cancelled", error="Task was cancelled during processing")
                return
                
            if html and isinstance(html, str) and len(html) > 0:
                cleaned_text = clean_text(html)
                if cleaned_text.strip():
                    # Improved chunking parameters
                    chunks = create_chunks(
                        cleaned_text, 
                        chunk_size=50,  # Increased from 64 to 512 for better context
                        overlap=10      # Increased overlap for better continuity
                    )
                    
                    # Add source URL to each chunk for better tracking
                    for chunk in chunks:
                        if chunk.strip():  # Only add non-empty chunks
                            chunk_with_source = f"Source: {url_page}\n\n{chunk}"
                            all_chunks.append(chunk_with_source)
                    
                    # Update progress periodically during processing
                    if i % 5 == 0:  # Update every 5 pages
                        update_progress(task_id, "processing", 
                                      pages_scraped=len(pages), 
                                      chunks_created=len(all_chunks))
        
        if not all_chunks:
            update_progress(task_id, "error", error="No valid text content found to ingest from the website")
            return
        
        # Filter out very short chunks that might not be useful
        filtered_chunks = [chunk for chunk in all_chunks if len(chunk.strip()) > 50]
        
        if not filtered_chunks:
            update_progress(task_id, "error", error="No substantial text content found after filtering")
            return
        
        logger.info(f"Created {len(filtered_chunks)} chunks from {len(pages)} pages")
        
        # Check for cancellation
        if task_id in cancellation_flags and cancellation_flags[task_id].is_set():
            update_progress(task_id, "cancelled", error="Task was cancelled")
            return
        
        update_progress(task_id, "processing", chunks_created=len(filtered_chunks))
        update_progress(task_id, "generating_embeddings")
        
        # Generate embeddings in batches to allow cancellation checks
        embeddings = []
        batch_size = 20  # Increased batch size for better performance
        
        for i in range(0, len(filtered_chunks), batch_size):
            # Check for cancellation
            if task_id in cancellation_flags and cancellation_flags[task_id].is_set():
                update_progress(task_id, "cancelled", error="Task was cancelled during embedding generation")
                return
                
            batch = filtered_chunks[i:i + batch_size]
            batch_embeddings = await asyncio.get_event_loop().run_in_executor(
                None, get_embeddings, batch
            )
            embeddings.extend(batch_embeddings)
            
            # Update progress
            progress_percent = ((i + batch_size) / len(filtered_chunks)) * 100
            update_progress(task_id, f"generating_embeddings ({progress_percent:.1f}%)", 
                          chunks_created=len(filtered_chunks))
        
        # Check for cancellation
        if task_id in cancellation_flags and cancellation_flags[task_id].is_set():
            update_progress(task_id, "cancelled", error="Task was cancelled")
            return
        
        update_progress(task_id, "storing")
        
        # Create progress callback for ingestion
        def ingestion_progress_callback(current: int, total: int, message: str):
            update_progress(task_id, "storing", 
                          chunks_created=len(filtered_chunks),
                          chunks_stored=current,
                          storage_progress=f"{current}/{total}")
        
        # Use the incremental ingestion function with progress tracking
        ingestion_result = await asyncio.get_event_loop().run_in_executor(
            None, 
            ingest_to_qdrant_incremental, 
            collection_name, 
            filtered_chunks, 
            embeddings, 
            task_id,
            ingestion_progress_callback
        )
        
        # Final check for cancellation
        if task_id in cancellation_flags and cancellation_flags[task_id].is_set():
            update_progress(task_id, "cancelled", error="Task was cancelled")
            return
        
        # Update final status based on ingestion result
        if ingestion_result["status"] == "completed":
            update_progress(task_id, "completed", 
                           result={
                               "collection_name": collection_name,
                               "pages_scraped": len(pages),
                               "chunks_created": len(filtered_chunks),
                               "chunks_stored": ingestion_result["ingested_points"],
                               "ingestion_status": ingestion_result["status"]
                           })
        elif ingestion_result["status"] == "partial":
            update_progress(task_id, "completed_with_warnings", 
                           result={
                               "collection_name": collection_name,
                               "pages_scraped": len(pages),
                               "chunks_created": len(filtered_chunks),
                               "chunks_stored": ingestion_result["ingested_points"],
                               "ingestion_status": "partial",
                               "warning": f"Only {ingestion_result['ingested_points']}/{ingestion_result['total_points']} chunks were stored"
                           })
        else:
            update_progress(task_id, "error", 
                           error=f"Ingestion failed with status: {ingestion_result['status']}")
        
    except asyncio.CancelledError:
        logger.info(f"Scraping task {task_id} was cancelled")
        update_progress(task_id, "cancelled", error="Task was cancelled")
        raise
    except Exception as e:
        logger.error(f"Error in scraping process: {e}")
        update_progress(task_id, "error", error=str(e))

def crawl_website_with_cancellation(url: str, task_id: str, max_pages=None):
    """Modified crawl_website function that checks for cancellation."""
    try:
        logger.info(f"Starting crawl for {url} with task_id {task_id}")
        
        # Check for cancellation before starting
        if task_id in cancellation_flags and cancellation_flags[task_id].is_set():
            logger.info(f"Crawling cancelled for task {task_id} before starting")
            return {}
        
        # Call your existing crawl_website function but with periodic cancellation checks
        # You should modify your existing crawl_website function to accept a cancellation callback
        
        # For now, this is a wrapper that calls your existing function
        # You should integrate cancellation checks into your actual crawl_website implementation
        result = crawl_website(url, max_pages)
        
        # Check for cancellation after crawling
        if task_id in cancellation_flags and cancellation_flags[task_id].is_set():
            logger.info(f"Crawling cancelled for task {task_id} after completion")
            return {}
        
        logger.info(f"Crawled {len(result)} pages for task {task_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error in crawl_website_with_cancellation: {e}")
        return {}

def cleanup_task(task_id: str):
    """Clean up completed task from running_tasks and cancellation flags."""
    if task_id in running_tasks:
        del running_tasks[task_id]
    if task_id in cancellation_flags:
        del cancellation_flags[task_id]
    logger.info(f"Cleaned up task {task_id}")

@router.delete("/scraping-tasks/{task_id}")
async def delete_scraping_task(
    task_id: str,
    current_user = Depends(get_current_active_user)
):
    """Delete a scraping task from memory (only completed/cancelled tasks)."""
    try:
        if task_id not in scraping_progress:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task_info = scraping_progress[task_id]
        
        # Only allow deletion of completed/cancelled tasks
        if not task_info["is_completed"]:
            raise HTTPException(
                status_code=400, 
                detail="Cannot delete active task. Stop the task first."
            )
        
        # Remove from progress tracking
        del scraping_progress[task_id]
        
        # Clean up running task and cancellation flag if exists
        if task_id in running_tasks:
            del running_tasks[task_id]
        if task_id in cancellation_flags:
            del cancellation_flags[task_id]
        
        return {
            "status": "deleted",
            "message": "Task deleted successfully",
            "task_id": task_id
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error deleting task {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))