"""
Agentic Orchestrator — Samsung Agentic RAG
Implements tool-based reasoning: the agent dynamically selects tools
(retrieval, OCR, STT) and orchestrates multi-step workflows.
"""

import os
import json
import re
from cerebras.cloud.sdk import Cerebras
from typing import Dict, List, Optional
from .retriever import HybridRetriever


class SupportAgent:
    """
    AI Agent that orchestrates:
    1. Query understanding and intent classification
    2. Tool selection (retrieval, comparison, category search)
    3. Context aggregation from multiple sources
    4. LLM response generation with strict grounding
    5. Confidence scoring and source citation
    """

    SYSTEM_BASE = """You are an advanced Samsung Support AI Assistant.
    Provide precise answers based ONLY on the provided context (Official Samsung Manuals OR samsung.com Web Results).

    STRICT FORMATTING AND GROUNDING RULES:
    1. NO HALLUCINATION: If the information is not in the RETRIEVED CONTEXT, state "I couldn't find information for this in our local manuals" and ask the user if they would like you to conduct a web search.
    2. NO SUBSTITUTION: Do not provide specs for a similar model if the user asked for a different one.
    3. PROFESSIONAL STYLE: Avoid all fluff, introductory filler, and emojis.
    4. STRUCTURE: Use clean Markdown tables for comparisons and bullet points for specifications.
    5. CITATION: List sources clearly: "Sources: [Official Catalog or samsung.com URL]".
    6. NO METADATA: Do not include confidence scores or internal labels in the response.
    7. WEB SEARCH: If confirmed, use ONLY Official Samsung Web Results (samsung.com) to answer. ABSOLUTELY NO non-Samsung sites.
    8. NO PRICING: If price is not in the context, state "Official pricing not available in this manual."
    9. NO TECH NOTES: DO NOT include any "Note:..." or explanations about retrieved context. Follow rules 1-8 strictly. """

    INTENT_PROMPT = """Classify the user's intent into exactly ONE of these categories:
- PRODUCT_INFO: User wants specifications, features, or details about a Samsung product
- COMPARISON: User wants to compare two or more Samsung products
- TROUBLESHOOTING: User has a problem/issue with their device
- CATEGORY_BROWSE: User wants to explore products in a category
- WEB_SEARCH_CONFIRM: User says "Yes", "Sure", "Go ahead", or similar to confirm a web search
- IMAGE_ANALYSIS: User uploaded an image
- GREETING: Hello, hi, thanks, etc.
- OUT_OF_SCOPE: Non-Samsung topics

Query: {query}
Intent:"""

    def __init__(self, api_key: str = None, retriever: HybridRetriever = None):
        self.api_key = api_key or os.environ.get("CEREBRAS_API_KEY")
        if not self.api_key:
            raise ValueError("CEREBRAS_API_KEY is missing.")
        self.client = Cerebras(api_key=self.api_key)
        self.retriever = retriever
        self.model = "llama3.1-8b"

    def classify_intent(self, query: str, last_bot_msg: str = "") -> str:
        """Uses LLM to classify user intent for smart routing."""
        # Special case: Confirmation for web search
        if last_bot_msg and "web search" in last_bot_msg.lower():
            if re.search(r'\b(yes|yeah|sure|yup|go ahead|ok|okay|do it)\b', query.lower()):
                return "WEB_SEARCH_CONFIRM"

        try:
            response = self.client.chat.completions.create(
                model="llama3.1-8b",
                messages=[
                    {"role": "system", "content": self.INTENT_PROMPT},
                    {"role": "user", "content": query},
                ],
                max_tokens=20,
                temperature=0.1,
            )
            intent = response.choices[0].message.content.strip().upper()
            
            # Normalize
            for vi in [
                "PRODUCT_INFO", "COMPARISON", "TROUBLESHOOTING", "WEB_SEARCH_CONFIRM",
                "CATEGORY_BROWSE", "IMAGE_ANALYSIS", "VOICE_QUERY",
                "GREETING", "OUT_OF_SCOPE"
            ]:
                if vi in intent:
                    return vi
            return "PRODUCT_INFO"
        except Exception:
            return "PRODUCT_INFO"

    def _web_search(self, query: str) -> str:
        """Simple web search fallback using DuckDuckGo."""
        try:
            import requests
            from bs4 import BeautifulSoup
            url = f"https://duckduckgo.com/html/?q=site:samsung.com+samsung+{query}"
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                results = soup.find_all('a', class_='result__a', limit=3)
                snippets = soup.find_all('div', class_='result__snippet', limit=3)
                
                output = f"WEB SEARCH RESULTS FOR '{query}':\n\n"
                for i, (r, s) in enumerate(zip(results, snippets)):
                    output += f"Source {i+1}: {r.get_text()} ({r.get('href')})\nSnippet: {s.get_text()}\n\n"
                return output if results else ""
        except Exception as e:
            print(f"WEB SEARCH ERROR: {e}")
        return ""

    def generate_response(self, user_query: str, chat_history: list,
                          input_type: str = "text",
                          image_analysis: str = None) -> Dict:
        """
        Main agentic pipeline:
        1. Classify intent
        2. Detect if web search confirmation
        3. Select and execute tools
        4. Aggregate context (Local or Web)
        5. Generate grounded response
        """
        # Get last bot message for state tracking
        last_bot_msg = ""
        for m in reversed(chat_history):
            if m["role"] == "assistant":
                last_bot_msg = m["content"]
                break

        # Step 1: Intent Classification
        intent = self.classify_intent(user_query, last_bot_msg)
        if image_analysis:
            intent = "IMAGE_ANALYSIS"

        # Step 2: Tool Selection & Execution
        retrieved_context = ""
        sources = []
        retrieval_scores = []
        is_relevant = False
        tool_used = "none"
        effective_query = user_query

        # 2a. Handle Web Search Confirmation
        if intent == "WEB_SEARCH_CONFIRM":
            search_query = user_query
            for i in range(len(chat_history) - 1, -1, -1):
                msg = chat_history[i]
                if msg["role"] == "assistant" and "web search" in msg["content"].lower():
                    if i > 0:
                        search_query = chat_history[i-1]["content"]
                    break
            
            retrieved_context = self._web_search(search_query)
            if retrieved_context:
                is_relevant = True
                tool_used = "web_search_duckduckgo"
                sources = ["Web Search Results"]
            else:
                yield {"type": "chunk", "text": "I tried searching the web but couldn't find a definitive answer. Is there anything else I can help with?"}
                return

        elif intent == "GREETING":
            # No retrieval needed for greetings
            tool_used = "greeting_handler"
        elif intent == "OUT_OF_SCOPE":
            yield {"type": "chunk", "text": "I specialize in Samsung products and services. For other topics, please consult a general AI assistant."}
            return
        elif intent in ["PRODUCT_INFO", "TROUBLESHOOTING", "IMAGE_ANALYSIS",
                         "VOICE_QUERY", "COMPARISON"]:
            # Use hybrid retriever
            if self.retriever:
                result = self.retriever.retrieve(effective_query)
                
                # FALLBACK: If nothing found, try a simplified query
                if not result["is_relevant"]:
                    simplified = self._simplify_query(user_query)
                    if simplified != user_query:
                        print(f"DEBUG: No results for '{user_query}'. Retrying with '{simplified}'...")
                        result = self.retriever.retrieve(simplified)
                
                retrieved_context = result["context"]
                sources = result["sources"]
                retrieval_scores = result.get("scores", [])
                is_relevant = result["is_relevant"]
                tool_used = f"hybrid_retriever (vector + graph)"

                # For comparisons, try a second retrieval if query mentions multiple products
                if intent == "COMPARISON" and is_relevant:
                    # Try to find the second product
                    words = user_query.lower().split()
                    comparison_keywords = ["vs", "versus", "compare", "or", "between", "and"]
                    for kw in comparison_keywords:
                        if kw in words:
                            idx = words.index(kw)
                            # Get text after comparison keyword
                            second_query = " ".join(words[idx+1:])
                            if second_query:
                                result2 = self.retriever.retrieve(second_query)
                                if result2["is_relevant"]:
                                    retrieved_context += "\n\n" + result2["context"]
                                    sources.extend(result2["sources"])
                                    sources = list(set(sources))
                                    tool_used = "comparison_retriever (dual search)"
                            break
                
                # RELEVANCE CHECK: If still not relevant, ask for web search permission
                if not is_relevant or not retrieved_context.strip():
                    yield {"type": "chunk", "text": "I couldn't find that specific information in our local Samsung manuals. Would you like me to conduct a web search on samsung.com to find more details?"}
                    return

        elif intent == "CATEGORY_BROWSE":
            # Extract category from query and do filtered search
            if self.retriever:
                category = self._extract_category(user_query)
                if category:
                    result = self.retriever.search_by_category(category, user_query)
                    retrieved_context = result["context"]
                    sources = result["sources"]
                    is_relevant = result["is_relevant"]
                    tool_used = f"category_search ({category})"
                else:
                    result = self.retriever.retrieve(user_query)
                    retrieved_context = result["context"]
                    sources = result["sources"]
                    is_relevant = result["is_relevant"]
                    tool_used = "hybrid_retriever (fallback)"
                
                if not is_relevant:
                    yield {"type": "chunk", "text": "I couldn't find any products in our local database matching that request. Should I try a web search?"}
                    return

        # Step 3: Build System Prompt
        system_prompt = self.SYSTEM_BASE + f"""

CURRENT INTENT: {intent}
TOOL USED: {tool_used}
INPUT TYPE: {input_type}

RETRIEVED CONTEXT:
{retrieved_context if retrieved_context else "No relevant knowledge found in the database."}
"""

        if image_analysis:
            system_prompt += f"\nIMAGE ANALYSIS RESULT:\n{image_analysis}\n"

        # Step 4: Build Messages
        messages = [{"role": "system", "content": system_prompt}]

        # Add recent chat history (last 6 turns for context)
        recent = chat_history[-6:] if len(chat_history) > 6 else chat_history
        for msg in recent:
            content = msg.get("content", "")
            if content:
                messages.append({
                    "role": msg["role"],
                    "content": content[:500]  # Truncate to manage context window
                })

        messages.append({"role": "user", "content": user_query})

        # Step 5: LLM Generation
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.15,
                max_tokens=1500,
                top_p=0.9,
                stream=True,
            )

            full_answer = ""
            for chunk in response:
                # print(f"DEBUG: Chunk: {chunk}")
                if hasattr(chunk, 'choices') and chunk.choices:
                    content = chunk.choices[0].delta.content
                    if content is not None:
                        full_answer += content
                        yield {"type": "chunk", "text": content}

            # print(f"DEBUG: Final Full Answer: {full_answer}")
            yield {
                "type": "metadata",
                "intent": intent,
                "tool_used": tool_used,
                "sources": sources,
                "is_relevant": is_relevant,
                "confidence": self._extract_confidence(full_answer),
            }

        except Exception as e:
            yield {
                "type": "error",
                "text": f"⚠️ I encountered an error while processing your request: {str(e)}. Please try again."
            }

    def _extract_confidence(self, answer: str) -> str:
        """Extracts confidence score from the LLM response."""
        patterns = [
            r'Confidence:\s*(\d+%)',
            r'🎯\s*Confidence:\s*(\d+%)',
            r'confidence[:\s]+(\d+%)',
        ]
        for pattern in patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                return match.group(1)
        return "N/A"

    def _extract_category(self, query: str) -> Optional[str]:
        """Extracts product category from a user query."""
        category_map = {
            "phone": "Smartphones", "smartphone": "Smartphones", "galaxy": "Smartphones",
            "mobile": "Smartphones",
            "tv": "Tvs", "television": "Tvs", "qled": "Tvs", "oled": "Tvs",
            "neo qled": "Tvs", "crystal uhd": "Tvs",
            "refrigerator": "Refrigerators", "fridge": "Refrigerators",
            "ac": "Air Conditioners", "air conditioner": "Air Conditioners",
            "windfree": "Air Conditioners",
            "washer": "Washers And Dryers", "dryer": "Washers And Dryers",
            "washing machine": "Washers And Dryers",
            "watch": "Watches", "galaxy watch": "Watches",
            "tablet": "Tablets", "tab": "Tablets",
            "monitor": "Monitors",
            "soundbar": "Audio Sound", "speaker": "Audio Sound",
            "earbuds": "Audio Sound", "buds": "Audio Sound",
            "dishwasher": "Dishwashers",
            "projector": "Projectors",
        }

        query_lower = query.lower()
        for keyword, category in category_map.items():
            if keyword in query_lower:
                return category
    def _simplify_query(self, query: str) -> str:
        """Strips common fluff from query to improve RAG matching."""
        stop_phrases = [
            "how to", "i want to", "please tell me", "do you know",
            "can you", "what is", "where is", "help me with", "show me"
        ]
        q = query.lower()
        for phrase in stop_phrases:
            q = q.replace(phrase, "")
        return q.strip()
