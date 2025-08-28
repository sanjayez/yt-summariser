"""
Query Enhancement Prompts
"""

QUERY_ENHANCEMENT_PROMPT = """Analyze this search query and help improve it for finding relevant YouTube videos:
Query: "{original_query}"

Please provide:
1. Key concepts, in the order they are mentioned from the passed query (2-4 main topics)
2. Enhanced search queries (2-3 variations that would find better videos)  
3. Intent type (TUTORIAL, LOOKUP)

IMPORTANT: Return ONLY valid JSON without any markdown formatting, code blocks, or additional text. Do not wrap the response in ```json``` or any other formatting.

{{
    "concepts": ["concept1", "concept2"],
    "enhanced_queries": ["query1", "query2"],
    "intent_type": "TUTORIAL"
}}"""


def get_enhancement_prompt(original_query: str) -> str:
    return QUERY_ENHANCEMENT_PROMPT.format(original_query=original_query)
