"""Zero-shot chapter detection prompt for LLM analysis"""

CHAPTER_PROMPT = """You are analyzing a video transcript to identify chapter boundaries.

TASK:
Identify distinct chapters in the transcript, including Introduction, main content chapters, advertisements/sponsorships, and outros.

INPUT:
A full normalized transcript (whitespace normalized, preserving original case and punctuation).

REQUIREMENTS:
1. Return EXACT verbatim quotes from the transcript for start and end boundaries
2. Identify ALL segments including intro, core content, ads, and outros
3. Chapters should be contiguous and non-overlapping
4. Use descriptive 3-5 word titles for each chapter
5. Produce at most 15 chapters; merge very short filler into adjacent chapters

CONTENT TYPES:
- "intro": Introduction/welcome segment
- "core": Main content of the video
- "ad": Advertisement/sponsorship
- "outro": Ending/closing segment
- "filler": Transitions or off-topic content

OUTPUT FORMAT (JSON ONLY):
{
  "chapters": [
    {
      "chapter": "3-5 word descriptive title",
      "content_type": "intro|core|ad|outro|filler",
      "start_string": "exact verbatim quote from transcript marking chapter start",
      "end_string": "exact verbatim quote from transcript marking chapter end"
    }
  ]
}

Return ONLY valid JSON without markdown formatting or additional text.
"""
