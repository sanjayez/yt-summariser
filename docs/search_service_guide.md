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

## Usage Examples

### Basic Usage

```python
from topic.services.providers.scrapetube_provider import ScrapeTubeProvider
from topic.services.search_service import YouTubeSearchService

# Create provider and service
provider = ScrapeTubeProvider(max_results=5)
service = YouTubeSearchService(provider)

# Simple search - returns list of URLs
urls = service.search_simple("python tutorial", max_results=5)
print(f"Found {len(urls)} URLs:")
for url in urls:
    print(f"  {url}")
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

# Create provider
provider = ScrapeTubeProvider(max_results=5, timeout=30)

# Simple search
urls = provider.search("django tutorial")

# Search with metadata
results = provider.search_with_metadata("react tutorial")
for result in results:
    print(f"{result.title} - {result.url}")
```

### Using Factory Functions

```python
from topic.services.search_service import create_youtube_search_service, default_search_service

# Create new service instance
service = create_youtube_search_service()

# Use default global instance
urls = default_search_service.search_simple("python tutorial")
```

## Configuration

### Provider Configuration

```python
# Configure provider settings
provider = ScrapeTubeProvider(
    max_results=10,  # Maximum results to return
    timeout=45       # Request timeout in seconds
)
```

### Service Configuration

```python
# Create service with custom provider
service = YouTubeSearchService(custom_provider)

# Or configure provider later
service.configure_provider(new_provider)
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

## Health Checks

```python
# Check if service is healthy
if service.health_check():
    print("Service is healthy")
else:
    print("Service is not responding")
```

## Provider Information

```python
# Get provider details
provider_info = service.get_provider_info()
print(f"Provider: {provider_info['name']} v{provider_info['version']}")

# Get service details
service_info = service.get_service_info()
print(f"Service: {service_info['service_name']}")
```

## Validation

The service includes built-in query validation:

```python
# Validate queries
valid_queries = [
    "python tutorial",
    "machine learning basics",
    "django rest framework"
]

invalid_queries = [
    "",           # Empty query
    "a",          # Too short
    "x" * 200     # Too long
]

for query in valid_queries:
    is_valid = service.provider.validate_query(query)
    print(f"'{query}': {is_valid}")
```

## Integration with Video Processing

The search service can be integrated with the existing video processing pipeline:

```python
from topic.services.search_service import default_search_service
from video_processor.tasks import process_video

# Search for videos
urls = default_search_service.search_simple("python tutorial", max_results=5)

# Process each video
for url in urls:
    print(f"Processing: {url}")
    # Use existing video processing pipeline
    process_video.delay(url)
```

## Best Practices

1. **Use the service layer**: Prefer `YouTubeSearchService` over direct provider usage
2. **Handle errors gracefully**: Always wrap search calls in try-catch blocks
3. **Validate queries**: Use the built-in validation before making searches
4. **Check health**: Verify service health before critical operations
5. **Respect rate limits**: The provider includes delays between requests
6. **Use metadata wisely**: Only request metadata when needed to improve performance

## Performance Considerations

- Search operations are synchronous and may take 1-3 seconds
- Results are limited to 5 by default to balance speed and utility
- The provider includes built-in delays to respect YouTube's rate limits
- Metadata requests are slightly slower than URL-only searches

## Troubleshooting

### Common Issues

1. **Network timeouts**: Increase timeout in provider configuration
2. **No results found**: Check query spelling and try broader terms
3. **Service unhealthy**: Verify internet connection and scrapetube functionality
4. **Invalid URLs**: Check that returned URLs contain valid YouTube video IDs

### Debugging

Enable logging to see detailed information:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('topic.services')
```

## Running the Demo

A complete demo is available at `examples/search_demo.py`:

```bash
python examples/search_demo.py
```

This demo showcases all features of the search service and provides examples of proper usage.