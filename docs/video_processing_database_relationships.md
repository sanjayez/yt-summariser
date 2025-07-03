# Database Relationships in yt_summariser

## Overview
This document describes the database models and their relationships in the `yt_summariser` project. The main apps involved are `api` and `video_processor`.

## Design Philosophy
The project follows a parallel processing design where video metadata and transcript processing are treated as independent operations:
Each Row from URLRequestTable is mapped to a row in VideoMetadata and VideoTranscript respectively

1. **Independent Data Sources**:
   - Metadata is fetched from YouTube's API (title, views, etc.)
   - Transcript can come from either:
     - YouTube's caption API
     - Direct audio processing
   - These sources can be processed in parallel, optimizing performance

2. **Fault Tolerance**:
   - If YouTube API fails, transcript processing can still succeed
   - If transcript extraction fails, metadata is still available
   - Each process has its own status tracking

---

## Models

### 1. `URLRequestTable` (`api/models.py`)
Central model that initiates parallel processing tasks.

| Field        | Type           | Description                       |
|--------------|----------------|-----------------------------------|
| request_id   | UUIDField      | Unique identifier for the request |
| url          | URLField       | The YouTube video URL             |
| ip_address   | GenericIPAddressField | IP address of the requester |
| status       | CharField      | ['processing', 'failed', 'success'] |
| created_at   | DateTimeField  | Timestamp of creation             |

---

### 2. `VideoMetadata` (`video_processor/models.py`)
Stores metadata from YouTube API for a video.

| Field        | Type           | Description                       |
|--------------|----------------|-----------------------------------|
| url_request  | OneToOneField  | Links to `URLRequestTable`        |
| title        | CharField      | Video title                       |
| description  | TextField      | Video description                 |
| duration     | IntegerField   | Duration in seconds               |
| channel_name | CharField      | Channel name                      |
| view_count   | BigIntegerField| Number of views                   |
| status       | CharField      | ['processing', 'failed', 'success'] |
| created_at   | DateTimeField  | Timestamp of creation             |

---

### 3. `VideoTranscript` (`video_processor/models.py`)
Stores transcript data, which can be obtained independently of metadata.

| Field           | Type           | Description                       |
|-----------------|----------------|-----------------------------------|
| url_request     | OneToOneField  | Links to `URLRequestTable`        |
| transcript_text | TextField      | Transcript text                   |
| language        | CharField      | Transcript language (default 'en')|
| status          | CharField      | ['processing', 'failed', 'success'] |
| created_at      | DateTimeField  | Timestamp of creation             |

---

## Parallel Processing Design

```
                 ┌─────────────────┐
                 │ URLRequestTable │
                 │                 │
                 │ • request_id    │
                 │ • url           │
                 │ • status        │
                 └─────────────────┘
                         │
           ┌────────────┴────────────┐
           │                         │
           ▼                         ▼
┌──────────────────┐      ┌─────────────────┐
│  VideoMetadata   │      │ VideoTranscript  │
│  (YouTube API)   │      │ (Audio/Captions) │
│                  │      │                  │
│ • title          │      │ • transcript     │
│ • channel_name   │      │ • language       │
│ • status         │      │ • status         │
└──────────────────┘      └─────────────────┘
```

This design enables:
- Parallel processing of metadata and transcript
- Independent status tracking for each process
- Fault isolation between processes
- Future extensibility for audio processing

---

## Status Propagation

The parallel design affects status propagation:
- Each model (`VideoMetadata` and `VideoTranscript`) maintains its own status
- `URLRequestTable` status reflects the combined state:
  - `'success'` if both processes succeed
  - `'failed'` if either process fails
  - `'processing'` while either is still processing

---

## Example Usage

```python
# Accessing related data
url_request = URLRequestTable.objects.get(request_id='abc123')

# Independent checks for each process
if hasattr(url_request, 'video_metadata'):
    print(f"Metadata status: {url_request.video_metadata.status}")
    print(f"Title: {url_request.video_metadata.title}")

if hasattr(url_request, 'video_transcript'):
    print(f"Transcript status: {url_request.video_transcript.status}")
    print(f"Language: {url_request.video_transcript.language}")

# Both processes can run independently
# Even if metadata fails, transcript might still be available and vice versa
```

---

## Future Extensions

The parallel design makes it easy to add new processing capabilities:
- Audio-to-text generation when YouTube captions aren't available
- Multiple transcript sources (YouTube captions, audio processing)
- Additional metadata sources
- Each new process can run independently while maintaining its own status

---

*This document provides a high-level overview of the database relationships and parallel processing design in the yt_summariser project. For more details, see the model definitions in the codebase.* 