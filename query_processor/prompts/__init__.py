"""
Query Processor Prompts Package
Centralized prompt templates for LLM interactions
"""

from .enhancement_prompts import QUERY_ENHANCEMENT_PROMPT, get_enhancement_prompt

__all__ = ["QUERY_ENHANCEMENT_PROMPT", "get_enhancement_prompt"]
