
from datetime import datetime
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from typing import List, Dict, Any, Optional
import json
import logging

from app.utils.conversation import get_conversation_history, update_conversation_history

logger = logging.getLogger(__name__)


class DatabaseChatMessageHistory(BaseChatMessageHistory):
    """Custom chat message history that stores in your existing database."""
    
    def __init__(self, conversation_id: str, db_session):
        self.conversation_id = conversation_id
        self.db = db_session
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve messages from database and convert to LangChain format."""
        try:
            # Get conversation history from your existing function  
            history = get_conversation_history(self.db, self.conversation_id)
            messages = []
            
            for msg in history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg.get("content", "")))
            
            return messages
        except Exception as e:
            logger.error(f"Error retrieving messages: {e}")
            return []
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the database."""
        try:
            # Convert LangChain message to your format
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            content = message.content
            
            # Get current history and append new message
            current_history = get_conversation_history(self.db, self.conversation_id)
            current_history.append({"role": role, "content": content})
            
            # Update using your existing function
            update_conversation_history(self.db, self.conversation_id, current_history)
            
        except Exception as e:
            logger.error(f"Error adding message: {e}")
    
    def clear(self) -> None:
        """Clear all messages from the conversation."""
        try:
            update_conversation_history(self.db, self.conversation_id, [])
        except Exception as e:
            logger.error(f"Error clearing messages: {e}")

def get_conversation_memory(conversation_id: str, db_session, memory_type: str = "buffer_window"):
    """
    Get LangChain conversation memory instance.
    
    Args:
        conversation_id: Unique conversation identifier
        db_session: Database session
        memory_type: Type of memory ("buffer_window", "summary_buffer")
    """
    
    # Create custom chat history instance
    chat_history = DatabaseChatMessageHistory(conversation_id, db_session)
    
    if memory_type == "buffer_window":
        # Keeps last k interactions in memory
        memory = ConversationBufferWindowMemory(
            chat_memory=chat_history,
            k=5,  # Keep last 5 exchanges
            return_messages=True,
            input_key="question",
            output_key="response"
        )
    elif memory_type == "summary_buffer":
        # Summarizes older conversations, keeps recent ones
        memory = ConversationSummaryBufferMemory(
            chat_memory=chat_history,
            max_token_limit=1000,
            return_messages=True,
            input_key="question",
            output_key="response"
        )
    else:
        raise ValueError(f"Unsupported memory type: {memory_type}")
    
    return memory


def clear_conversation_memory(conversation_id: str, db_session):
    """Clear conversation memory for a specific conversation."""
    try:
        memory = get_conversation_memory(conversation_id, db_session)
        memory.clear()
        logger.info(f"Cleared conversation memory for ID: {conversation_id}")
    except Exception as e:
        logger.error(f"Error clearing conversation memory: {e}")

def get_conversation_summary(conversation_id: str, db_session) -> str:
    """Get a summary of the conversation using LangChain."""
    try:
        memory = get_conversation_memory(conversation_id, db_session, "summary_buffer")
        messages = memory.chat_memory.messages
        
        if not messages:
            return "No conversation history available."
        
        # Create a simple summary
        summary = f"Conversation with {len(messages)} messages. "
        if len(messages) >= 2:
            summary += f"Started with: '{messages[0].content[:50]}...'"
        
        return summary
    except Exception as e:
        logger.error(f"Error getting conversation summary: {e}")
        return "Unable to generate conversation summary."
    
    # Global in-memory storage for conversations
CONVERSATION_MEMORY: Dict[str, List[Dict]] = {}

def get_simple_conversation_history(collection_name: str) -> List[Dict]:
    """Get conversation history from simple in-memory storage."""
    return CONVERSATION_MEMORY.get(collection_name, [])

def add_to_conversation_history(collection_name: str, role: str, content: str):
    """Add message to conversation history."""
    if collection_name not in CONVERSATION_MEMORY:
        CONVERSATION_MEMORY[collection_name] = []
    
    CONVERSATION_MEMORY[collection_name].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    
    # Keep only last 10 messages to prevent memory bloat
    if len(CONVERSATION_MEMORY[collection_name]) > 10:
        CONVERSATION_MEMORY[collection_name] = CONVERSATION_MEMORY[collection_name][-10:]


def clear_conversation_history(collection_name: str):
    """Clear conversation history for a collection."""
    if collection_name in CONVERSATION_MEMORY:
        CONVERSATION_MEMORY[collection_name] = []