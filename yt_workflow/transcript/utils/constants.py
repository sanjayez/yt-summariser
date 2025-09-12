"""Constants for transcript chunking"""

# Micro chunk settings (in seconds)
MICRO_TARGET = 55.0  # Target duration for micro chunks
MICRO_MAX = 90.0  # Maximum duration for micro chunks
MICRO_OVERLAP = 8.0  # Overlap between consecutive micro chunks

# Macro chunk settings (in seconds)
MACRO_TARGET = 300.0  # Target duration for macro chunks (5 minutes)
MACRO_MAX = 360.0  # Maximum duration for macro chunks (6 minutes)

# Sentence ending punctuation
SENTENCE_ENDINGS = (".", "!", "?")
