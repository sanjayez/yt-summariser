# YouTube Search Service Guide

This guide explains how to use the YouTube Search Service and ScrapeTube Provider for searching YouTube videos.

## Overview

The YouTube Search Service provides a clean, abstract interface for searching YouTube videos using the `scrapetube` library. It follows the established patterns from the `ai_utils` module and provides proper error handling, logging, and validation.

## Components

### 1. ScrapeTubeProvider

The core provider that handles YouTube search functionality using the scrapetube library.

**Location**: `topic/services/providers/scrapetube_provider.py`

**Key Features**:
- Returns exactly 5 video URLs by default (configurable)
- Supports search with metadata (title, channel, duration, views, etc.)
- **NEW: English-only filtering** to exclude non-English videos
- **NEW: YouTube Shorts filtering** based on video duration
- **NEW: Customizable duration thresholds** for filtering
- Proper error handling and fallback mechanisms
- Query validation and health checks
- Configurable timeout and result limits

### 2. SearchService

An abstract service layer that provides dependency injection and follows the ai_utils interface patterns.

**Location**: `topic/services/search_service.py`

**Key Features**:
- Abstract `SearchProvider` interface
- `YouTubeSearchService` wrapper class
- Support for different search providers
- Structured request/response models
- Health checks and provider configuration

## Available scrapetube Filters

The underlying `scrapetube.get_search()` function supports these built-in filters:

1. **`sort_by`** - Sort order for results:
   - `"relevance"` (default) - By relevance to search query
   - `"upload_date"` - Newest videos first
   - `"view_count"` - Most popular videos first
   - `"rating"` - Videos with more likes first

2. **`results_type`** - Type of content to search for:
   - `"video"` (default) - Only videos
   - `"channel"` - Only channels
   - `"playlist"` - Only playlists
   - `"movie"` - Only movies

3. **`limit`** - Maximum number of results
4. **`sleep`** - Delay between requests (default: 1 second)

## Custom Filtering Features

Since scrapetube doesn't have built-in filters for language or video duration, we've implemented custom post-processing filters:

### English-Only Filtering
- Uses Unicode character range detection to identify non-English text
- Checks for Chinese, Japanese, Korean, Arabic, Russian, Hindi, Thai characters
- Validates ASCII character ratio (80% threshold)
- Configurable via `english_only` parameter

### YouTube Shorts Filtering
- Parses video duration to exclude short-form content
- Default threshold: 60 seconds minimum duration
- Supports duration formats: "1:23", "12:34", "1:23:45"
- Configurable via `filter_shorts` and `min_duration_seconds` parameters

## Usage Examples

### Basic Usage with Filtering

```python
from topic.services.providers.scrapetube_provider import ScrapeTubeProvider
from topic.services.search_service import YouTubeSearchService

# Create provider with English-only and shorts filtering
provider = ScrapeTubeProvider(
    max_results=5,
    timeout=30,
    filter_shorts=True,      # Filter out YouTube shorts
    english_only=True,       # Only English videos
    min_duration_seconds=60  # Videos must be at least 60 seconds
)
service = YouTubeSearchService(provider)

# Simple search - returns filtered list of URLs
urls = service.search_simple("python tutorial", max_results=5)
print(f"Found {len(urls)} filtered URLs:")
for url in urls:
    print(f"  {url}")
```

### Customizing Filter Settings

```python
# More permissive filtering
provider = ScrapeTubeProvider(
    max_results=10,
    filter_shorts=False,     # Allow shorts
    english_only=False,      # Allow all languages
    min_duration_seconds=30  # Shorter minimum duration
)

# Strict filtering for long-form English content
provider = ScrapeTubeProvider(
    max_results=5,
    filter_shorts=True,
    english_only=True,
    min_duration_seconds=300  # 5 minutes minimum
)
```

### Advanced Usage with Metadata

```python
from topic.services.search_service import SearchRequest

# Create detailed search request
request = SearchRequest(
    query="machine learning basics",
    max_results=3,
    include_metadata=True
)

# Execute search
response = service.search(request)

print(f"Query: {response.query}")
print(f"Total Results: {response.total_results}")
print(f"Search Time: {response.search_time_ms:.2f}ms")

# Access results
for url in response.results:
    print(f"URL: {url}")

# Access metadata (if requested)
if response.metadata:
    for result in response.metadata:
        print(f"Title: {result.title}")
        print(f"Channel: {result.channel_name}")
        print(f"Duration: {result.duration}")
        print(f"Views: {result.view_count}")
```

### Direct Provider Usage

```python
from topic.services.providers.scrapetube_provider import ScrapeTubeProvider

# Create provider with custom settings
provider = ScrapeTubeProvider(
    max_results=5, 
    timeout=30,
    filter_shorts=True,
    english_only=True,
    min_duration_seconds=120  # 2 minutes minimum
)

# Simple search
urls = provider.search("django tutorial")

# Search with metadata
results = provider.search_with_metadata("react tutorial")
for result in results:
    print(f"{result.title} ({result.duration}) - {result.url}")
```

### Using Factory Functions

```python
from topic.services.search_service import create_youtube_search_service, default_search_service

# Create new service instance with custom provider
custom_provider = ScrapeTubeProvider(
    filter_shorts=True,
    english_only=True
)
service = create_youtube_search_service(custom_provider)

# Use default global instance (uses default ScrapeTubeProvider settings)
urls = default_search_service.search_simple("python tutorial")
```

## Configuration

### Provider Configuration

```python
# Configure provider settings
provider = ScrapeTubeProvider(
    max_results=10,              # Maximum results to return
    timeout=45,                  # Request timeout in seconds
    filter_shorts=True,          # Filter out YouTube shorts
    english_only=True,           # Only English content
    min_duration_seconds=120     # Minimum video duration (2 minutes)
)
```

### Service Configuration

```python
# Create service with custom provider
service = YouTubeSearchService(custom_provider)

# Or configure provider later
service.configure_provider(new_provider)
```

## Filter Configuration Examples

### Production Configuration (Recommended)
```python
# Optimized for quality English content, no shorts
provider = ScrapeTubeProvider(
    max_results=5,
    filter_shorts=True,
    english_only=True,
    min_duration_seconds=60
)
```

### Development/Testing Configuration
```python
# More permissive for testing
provider = ScrapeTubeProvider(
    max_results=10,
    filter_shorts=False,
    english_only=False,
    min_duration_seconds=10
)
```

### Long-form Content Configuration
```python
# For educational/tutorial content
provider = ScrapeTubeProvider(
    max_results=5,
    filter_shorts=True,
    english_only=True,
    min_duration_seconds=300  # 5 minutes minimum
)
```

## Error Handling

The service includes comprehensive error handling:

```python
try:
    urls = service.search_simple("invalid query")
except ValueError as e:
    print(f"Invalid query: {e}")
except Exception as e:
    print(f"Search failed: {e}")
```

## Troubleshooting

### Common Issues

1. **Few or no results after filtering**: Try relaxing filter settings
   - Set `filter_shorts=False` to include shorts
   - Set `english_only=False` to include all languages
   - Reduce `min_duration_seconds` threshold

2. **Network timeouts**: Increase timeout in provider configuration
3. **No results found**: Check query spelling and try broader terms
4. **Service unhealthy**: Verify internet connection and scrapetube functionality
5. **Invalid URLs**: Check that returned URLs contain valid YouTube video IDs

### Filter Debugging

Enable debug logging to see what content is being filtered:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('topic.services')

# You'll see messages like:
# "Filtering out short video: 0:45 (45s)"
# "Filtering out non-English video: 你好世界"
```

### Performance Considerations

- **Iterative fetching**: Keeps searching until finding the required number of filtered results
- **Safety limits**: Maximum 100 total videos checked to prevent infinite loops
- **Smart stopping**: Stops immediately when target number of filtered results is found
- English detection is lightweight but not perfect
- Duration parsing is fast and reliable
- Consider caching results for repeated queries

## Running the Demo

A complete demo is available at `examples/search_demo.py`:

```bash
python examples/search_demo.py
```

This demo showcases all features including the new filtering capabilities.