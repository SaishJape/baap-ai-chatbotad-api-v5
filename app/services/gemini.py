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

# def translate_to_english(user_query: str) -> str:
#     model = genai.GenerativeModel("gemini-2.0-flash")
#     prompt = f"Translate the following into English:\n\n{user_query}"
#     response = model.generate_content(prompt)
#     return response.text.strip()

import google.generativeai as genai

# Make sure to configure your API key before calling the function, e.g.:
genai.configure(api_key="AIzaSyCo2Hhvv_Qs1O52jGj7EMXL1Ve4HgaOLyM")

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


def ask_gemini(context: str, question: str, query_analysis: dict, enhanced_results: dict, conversation_history: Optional[List[Dict[str, str]]] = None) -> dict:
    """Ask Gemini and return a structured JSON response with optional buttons."""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Format conversation history if available
        conversation_context = ""
        if conversation_history:
            conversation_context = "Previous conversation:\n"
            for msg in conversation_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                conversation_context += f"{role.capitalize()}: {content}\n"
            conversation_context += "\n"

        prompt = (
            "You are the official AI assistant of the company. Your personality is smart, professional, and friendly — but most importantly, you must always be **easy for users to understand**.\n"
            "You are designed to handle **any kind of user message**, even if it is unrelated, unclear, or confusing. Your job is to respond in a helpful, polite, and clearly structured way.\n"
            "Use the internal company content when relevant, but always provide a meaningful response regardless of context.\n\n"

            "🎯 Output Format Instructions:\n"
            "- ONLY return a **valid raw JSON object**. Do NOT include markdown (```json), extra quotes, or any surrounding text.\n"
            "- The JSON must contain exactly the following 4 keys:\n"
            "  1. 'response': string → A helpful, easy-to-understand answer. Use friendly and clear language. If relevant context is available, use it. If not, still give a meaningful response. Format the response with:\n"
            "     - `\\n` for line breaks or separate thoughts\n"
            "     - `**...**` to highlight important words or phrases\n"
            "     - Keep it simple and conversational.\n"
            "  2. 'buttons': boolean → true **only** if actionable info like email, phone, or LinkedIn is present **and relevant**.\n"
            "  3. 'button_type': list of strings like [\"email\", \"linkedin\", \"website\", \"phone\"], or null if buttons is false.\n"
            "  4. 'button_data': list of actual values from the context, or null if buttons is false.\n\n"

            "🧠 Rules:\n"
            "- Be warm and friendly — even if the question is strange, off-topic, or confusing.\n"
            "- If the user greets you (e.g., 'hi', 'hello'), reply politely and cheerfully.\n"
            "- If the message is unclear or doesn't make sense, respond **politely** asking for clarification.\n"
            "- Never guess or invent data. Only use real information from `enhanced_results` for buttons.\n"
            "- Always return the full JSON structure, even if buttons are not needed. Use:\n"
            "  \"buttons\": false,\n"
            "  \"button_type\": null,\n"
            "  \"button_data\": null\n\n"

            "✅ Example 1 (greeting or general help):\n"
            '{\n'
            '  "response": "Hello!\\n\\nI\'m here to help you with anything related to our company. You can ask about **services**, **contacts**, **processes**, or anything else — and I\'ll do my best to assist you.",\n'
            '  "buttons": false,\n'
            '  "button_type": null,\n'
            '  "button_data": null\n'
            '}\n\n'

            "✅ Example 2 (clear question with contact info):\n"
            '{\n'
            '  "response": "Sure!\\n\\nYou can reach our **customer support team** via **email at support@company.com** or follow our updates on **LinkedIn**.\\n\\nLet me know if you need help with something specific!",\n'
            '  "buttons": true,\n'
            '  "button_type": ["email", "linkedin"],\n'
            '  "button_data": ["support@company.com", "https://linkedin.com/company/example"]\n'
            '}\n\n'

            "✅ Example 3 (user asks something confusing):\n"
            '{\n'
            '  "response": "Thanks for your message!\\n\\nI didn’t quite understand your question. Could you please rephrase or give a bit more detail? I’m here to help with anything related to our company.",\n'
            '  "buttons": false,\n'
            '  "button_type": null,\n'
            '  "button_data": null\n'
            '}\n\n'

            f"{conversation_context}"
            f"📄 Internal Company Content:\n{enhanced_results}\n\n"
            f"❓ User Message:\n{question}\n\n"
            "✍️ Please respond now with the final raw JSON object only:"
        )

        response = model.generate_content(prompt)
        text = response.text.strip()

        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown-like ```json block
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                cleaned_json = json_match.group()
                return json.loads(cleaned_json)
            else:
                logging.warning("Could not extract valid JSON from Gemini response.")
                return {
                    "response": "Sorry, I couldn't generate a valid response.",
                    "buttons": False,
                    "button_type": None,
                    "button_data": None
                }

    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return {
            "response": "Sorry, an error occurred while processing your request.",
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
        # Step 1: Process query with Gemini
        processed_query = process_query_with_gemini(user_query)
        logger.debug(f"Processed query: {processed_query}")

        # Step 2: Perform vector search in Qdrant
        search_results = query_qdrant(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
        logger.debug(f"Search results from Qdrant: {search_results}")

        # Step 3: Extract context text from search results
        context_chunks = []
        for result in search_results:
            if result.get("payload") and result["payload"].get("text"):
                score = result.get("score", 0)
                text = result["payload"]["text"]
                context_chunks.append(f"[Relevance: {score:.3f}] {text}")
        
        # print("context_chunks (context text from search results) : ", context_chunks)

        context_text = "\n\n".join(context_chunks)

        print("context_text (context text from search results) : ", context_text)

        return {
            "processed_query": processed_query,
            "search_results": search_results,
            "context_text": context_text
        }

    except Exception as e:
        logger.error(f"Error in enhanced_query_with_gemini: {e}")
        return {
            "error": str(e),
            "response": "I apologize, but I encountered an error while processing your query."
        }