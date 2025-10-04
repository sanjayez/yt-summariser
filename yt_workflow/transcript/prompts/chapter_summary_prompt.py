"""Prompt for summarizing chapter content into bullet points"""

CHAPTER_SUMMARY_PROMPT = """Summarize this chapter into exactly 3-4 concise bullet points focusing on key concepts and main ideas.
Return only JSON: {"title": "chapter title", "bullet_points": ["point1", "point2", "point3"]}
return - only title and bullet_points."""
