import json
from typing import Any, Dict, List, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging
import re

from app.db.qdrant import query_qdrant
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is required")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

genai.configure(api_key=api_key)

import google.generativeai as genai

# Make sure to configure your API key before calling the function, e.g.:
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def translate_to_english(user_query: str) -> str:
    """
    Translates the given text into English using the Gemini-2.0-flash model.

    This function aims to provide a clear and direct translation by instructing the
    model to return only the translated text, without any additional commentary,
    introductions, or formatting. It also includes error handling for robustness.

    Args:
        user_query (str): The text string to be translated into English.

    Returns:
        str: The clean, translated English text. Returns an empty string if
             translation fails or no text is returned by the model.
    """
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Refined prompt: Explicitly instruct the model to return only the translation.
    prompt = (
        "Translate the following text into English.\n"
        "Text may be in any language. \n"
        "Your response should be a direct translation without any additional commentary, introductions, or formatting.\n\n"
        "You are a professional translator. Your task is to provide a clear and accurate translation of the given text.\n\n"
        "Instructions:\n"
        "- Translate the text into English.\n"
        "- Do not include any additional text, explanations, or formatting.\n"
        "Provide ONLY the translated text, with no additional commentary, introductions, "
        "or formatting (e.g., 'Here is the translation:').\n\n"
        f"Text to translate: {user_query}"
    )

    try:
        response = model.generate_content(prompt)

        # Check if the response object and its 'text' attribute exist and are not empty
        if response and hasattr(response, 'text') and response.text:
            # Use .strip() to remove any leading/trailing whitespace just in case
            return response.text.strip()
        else:
            print(f"Warning: Gemini model returned an empty or invalid response for query: '{user_query}'")
            return "" # Return an empty string for no valid translation
    except Exception as e:
        # Catch any exceptions that might occur during the API call (e.g., network issues, API errors)
        print(f"Error during translation for query '{user_query}': {e}")
        return "" # Return an empty string on error


def ask_gemini_fast(
    context: str, 
    question: str, 
    query_analysis: dict, 
    enhanced_results: dict, 
    conversation_history: List[Dict]
) -> dict:
    """Ask Gemini with simple conversation history (ultra-fast)."""
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Build conversation context quickly
        conversation_context = ""
        is_first_message = len(conversation_history) == 0
        
        if not is_first_message and conversation_history:
            conversation_context = "Previous conversation:\n"
            # Only use last 6 messages (3 exchanges) to keep it fast
            recent_messages = conversation_history[-6:]
            
            for msg in recent_messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n"
            conversation_context += "\n"

        # Ensure context is not empty
        if not context or context.strip() == "":
            context = "No specific context available from internal documents."

        # Dynamic greeting logic (simplified)
        if is_first_message:
            greeting_instruction = (
                "🎯 FIRST MESSAGE RULES:\n"
                "- Simple greetings (hi, hello) → 'Hello! How can I help you today?'\n"
                "- Specific questions → Brief friendly acknowledgment + direct answer\n"
                "- Business questions → 'Hello! I'm here to help with information about our company.'\n\n"
            )
        else:
            greeting_instruction = (
                "🎯 FOLLOW-UP RULES:\n"
                "- NO greetings or introductions\n"
                "- Answer directly based on conversation context\n\n"
            )

        # Simplified prompt for faster processing
        prompt = (
            "You are the AI assistant for The Baap Company. Be smart, professional, and friendly.\n"
            "Respond helpfully to any message. Use company content when relevant.\n\n"

            f"{greeting_instruction}"

            "📋 OUTPUT FORMAT - Return ONLY valid JSON:\n"
            "{\n"
            '  "response": "Your helpful answer with **bold** formatting and \\n for breaks",\n'
            '  "buttons": true/false,\n'
            '  "button_type": ["email", "phone"] or null,\n'
            '  "button_data": ["actual values"] or null\n'
            "}\n\n"

            "🎨 FORMATTING RULES FOR RESPONSE:\n"
            "- Use **bold text** for important keywords, headings, and emphasis\n"
            "- Use \\n\\n for paragraph breaks (double newline)\n"
            "- Use \\n for single line breaks\n"
            "- Use numbered lists: 1. First item\\n2. Second item\\n3. Third item\n"
            "- Use bullet points: • First point\\n• Second point\\n• Third point\n"
            "- Use *italic text* for subtle emphasis or quotes\n"
            "- Use --- for horizontal dividers when separating sections\n"
            "- Use > for quotes or important notes\n"
            "- Use `code formatting` for technical terms or specific values\n"
            "- Use emojis appropriately: ✅ ❌ 📞 📧 🏢 💼 ⭐ 🎯 📋 💡\n"
            "- Structure long responses with clear sections and headings\n\n"

            f"{conversation_context}"
            f"📄 Company Content:\n{context}\n\n"
            f"❓ User Question: {question}\n\n"
            "JSON Response:"
        )

        print("conversation_context:", conversation_context)
        print("context:", context)

        # Generate response
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Quick JSON parsing
        try:
            # Try direct parse first
            parsed_response = json.loads(text)
            
            # Validate required keys
            required_keys = ['response', 'buttons', 'button_type', 'button_data']
            if all(key in parsed_response for key in required_keys):
                return parsed_response
                
        except json.JSONDecodeError:
            pass

        # Fallback: Extract JSON from text
        try:
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                cleaned_json = json_match.group()
                parsed_response = json.loads(cleaned_json)
                
                # Validate required keys
                required_keys = ['response', 'buttons', 'button_type', 'button_data']
                if all(key in parsed_response for key in required_keys):
                    return parsed_response
        except:
            pass
        
        # Final fallback
        logger.warning("Could not parse Gemini JSON response, using fallback")
        return {
            "response": "I apologize, but I'm having trouble processing your request right now. Please try rephrasing your question.",
            "buttons": False,
            "button_type": None,
            "button_data": None
        }

    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return {
            "response": "Sorry, an error occurred while processing your request. Please try again later.",
            "buttons": False,
            "button_type": None,
            "button_data": None
        }

def analyze_user_query(question: str) -> dict:
    """Analyze user query to extract key information and intent."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        "Analyze this user question and extract key information:\n"
        "1. Main topic/subject\n"
        "2. Key keywords for search\n"
        "3. Question type (factual, how-to, definition, comparison, etc.)\n"
        "4. Intent (what specifically they want to know)\n\n"
        "Return response in this JSON format:\n"
        "{\n"
        '  "main_topic": "extracted main topic",\n'
        '  "keywords": ["keyword1", "keyword2", "keyword3"],\n'
        '  "question_type": "factual/how-to/definition/etc",\n'
        '  "intent": "specific intent description"\n'
        "}\n\n"
        f"User Question: {question}"
    )
    
    try:
        response = model.generate_content(prompt)
        import json
        return json.loads(response.text.strip())
    except:
        return {
            "main_topic": question,
            "keywords": [question],
            "question_type": "general",
            "intent": "general information"
        }

def process_query_with_gemini(user_query: str) -> Dict[str, Any]:
    """
    Process user query with Gemini to extract key information and generate search parameters.
    Translates query if needed.
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""
        You are a multilingual assistant. The user query might be in any language.

        Step 1: Translate the query to English if it's not already.
        Step 2: Extract key information for database search.

        Query: {user_query}

        Output this JSON structure:
        {{
          "search_terms": [ ... ],
          "requirements": [ ... ],
          "context": "..." 
        }}
        """

        response = model.generate_content(prompt)
        processed_text = response.text.strip()
        print("Gemini raw response:", processed_text)

        # Clean ```json block if present
        match = re.search(r'\{.*\}', processed_text, re.DOTALL)
        if match:
            cleaned_json = match.group(0)
            search_params = json.loads(cleaned_json)
        else:
            raise ValueError("No valid JSON object found.")

        print("Parsed search parameters:", search_params)
        return search_params

    except Exception as e:
        logger.error(f"Error processing query with Gemini: {e}")
        return {
            "search_terms": [user_query],
            "requirements": [],
            "context": ""
        }
    
def enhanced_query_with_gemini(
    collection_name: str,
    user_query: str,
    query_vector: List[float],
    limit: int = 10
) -> Dict[str, Any]:
    """
    Enhanced query process that uses Gemini for query understanding and response generation.
    """
    try:
        # Input validation
        if not collection_name or not user_query or not query_vector:
            raise ValueError("Missing required parameters")

        # Step 1: Process query with Gemini
        try:
            processed_query = process_query_with_gemini(user_query)
            logger.debug(f"Processed query: {processed_query}")
        except Exception as e:
            logger.warning(f"Query processing failed, using original query: {e}")
            processed_query = {"original_query": user_query}

        # Step 2: Perform vector search in Qdrant
        try:
            search_results = query_qdrant(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit
            )
            logger.debug(f"Found {len(search_results)} search results from Qdrant")
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            search_results = []

        # Step 3: Extract and validate context text from search results
        context_chunks = []
        if search_results:
            for i, result in enumerate(search_results):
                try:
                    # Handle different result structures
                    payload = result.get("payload", {})
                    if isinstance(payload, dict):
                        text = payload.get("text", "")
                    else:
                        text = str(payload) if payload else ""
                    
                    if text and text.strip():
                        score = result.get("score", 0.0)
                        context_chunks.append(f"[Relevance: {score:.3f}] {text.strip()}")
                        
                except Exception as e:
                    logger.warning(f"Error processing search result {i}: {e}")
                    continue

        # Create context text
        if context_chunks:
            context_text = "\n\n".join(context_chunks)
            logger.info(f"Generated context with {len(context_chunks)} chunks")
        else:
            context_text = "No relevant context found in the knowledge base."
            logger.warning("No valid context chunks found")

        print(f"Final context_text: {context_text[:500]}...")  # Print first 500 chars for debugging

        return {
            "processed_query": processed_query,
            "search_results": search_results,
            "context_text": context_text,
            "total_results": len(search_results),
            "context_chunks_count": len(context_chunks)
        }

    except Exception as e:
        logger.error(f"Error in enhanced_query_with_gemini: {e}")
        return {
            "error": str(e),
            "processed_query": {"error": "Query processing failed"},
            "search_results": [],
            "context_text": "Error occurred while retrieving context.",
            "total_results": 0,
            "context_chunks_count": 0
        }