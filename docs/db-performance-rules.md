# 🚀 **Database CRUD Efficiency Guidelines**

Based on your current database structure, here are specific patterns to minimize reads/writes:

## 📖 **READ Operations**

### **1. Use select_related for Forward Relationships**
```python
# ❌ BAD: N+1 queries (1 + N queries for N URL requests)
url_requests = URLRequestTable.objects.all()
for url_req in url_requests:
    print(url_req.search_request.original_query)  # Extra query each time!

# ✅ GOOD: Single JOIN query
url_requests = URLRequestTable.objects.select_related('search_request').all()
for url_req in url_requests:
    print(url_req.search_request.original_query)  # No extra queries
```

### **2. Use prefetch_related for Reverse Relationships**
```python
# ❌ BAD: N+1 queries
search_requests = SearchRequest.objects.all()
for search in search_requests:
    print(search.url_requests.count())  # Extra query each time!

# ✅ GOOD: 2 queries total (1 + 1 prefetch)
search_requests = SearchRequest.objects.prefetch_related('url_requests').all()
for search in search_requests:
    print(search.url_requests.count())  # Uses prefetched data
```

### **3. Efficient Deep Relationships**
```python
# ✅ Get complete search-to-segments pipeline efficiently
search_requests = SearchRequest.objects.select_related(
    'search_session'  # Forward FK
).prefetch_related(
    'url_requests__video_metadata__video_transcript__segments'  # Deep reverse
).all()

# Now you can traverse without extra queries:
for search in search_requests:
    session_ip = search.search_session.user_ip  # No extra query
    for url_req in search.url_requests.all():   # No extra query
        video = url_req.video_metadata           # No extra query
        segments = video.video_transcript.segments.all()  # No extra query
```

### **4. Use Natural Keys for Direct Access**
```python
# ✅ Direct access using natural keys (single query)
video = VideoMetadata.objects.get(video_id='dQw4w9WgXcQ')
transcript = VideoTranscript.objects.get(video_id='dQw4w9WgXcQ')
segment = TranscriptSegment.objects.get(segment_id='dQw4w9WgXcQ_001')

# ✅ Bulk natural key queries
video_ids = ['vid1', 'vid2', 'vid3']
videos = VideoMetadata.objects.filter(video_id__in=video_ids)
```

### **5. Use values() for Lightweight Data**
```python
# ❌ BAD: Fetches full objects when you only need specific fields
videos = VideoMetadata.objects.all()
titles = [v.title for v in videos]  # Loads all fields

# ✅ GOOD: Only fetch needed fields
titles = VideoMetadata.objects.values_list('title', flat=True)
```

## ✏️ **CREATE Operations**

### **1. Batch Creation with bulk_create**
```python
# ✅ Create multiple URL requests efficiently
url_requests = [
    URLRequestTable(
        search_request=search_request,
        url=url,
        ip_address=ip_address
    ) for url in video_urls
]
URLRequestTable.objects.bulk_create(url_requests)
```

### **2. Use get_or_create for Avoiding Duplicates**
```python
# ✅ Prevent duplicate videos from different searches
video, created = VideoMetadata.objects.get_or_create(
    video_id=youtube_video_id,
    defaults={
        'url_request': url_request,
        'title': title,
        'description': description,
        # ... other fields
    }
)

if not created:
    # Video exists, just link it to this URL request
    # Handle business logic for existing video
    pass
```

### **3. Efficient Segment Creation**
```python
# ✅ Bulk create segments with proper segment_id
segments = []
for i, segment_data in enumerate(transcript_segments, 1):
    segments.append(TranscriptSegment(
        segment_id=f"{video_id}_{i:03d}",  # Natural key format
        transcript=transcript,
        sequence_number=i,
        start_time=segment_data['start'],
        duration=segment_data['duration'],
        text=segment_data['text']
    ))

TranscriptSegment.objects.bulk_create(segments)
```

## 🔄 **UPDATE Operations**

### **1. Bulk Updates Without Fetching Objects**
```python
# ✅ Update status for all URL requests in a search (single query)
URLRequestTable.objects.filter(
    search_request=search_request
).update(status='processing')

# ✅ Mark segments as embedded after vector store upload
TranscriptSegment.objects.filter(
    transcript__video_metadata__video_id__in=processed_video_ids
).update(is_embedded=True)
```

### **2. Conditional Updates**
```python
# ✅ Update only specific fields efficiently
VideoMetadata.objects.filter(
    video_id=video_id
).update(
    status='success',
    view_count=new_view_count,
    is_embedded=True
)
```

## 🗑️ **DELETE Operations**

### **1. Leverage CASCADE for Related Deletes**
```python
# ✅ Delete search session (cascades to all related data)
SearchSession.objects.filter(session_id=session_id).delete()
# Automatically deletes: SearchRequest → URLRequestTable → VideoMetadata → VideoTranscript → TranscriptSegment

# ✅ Delete specific video and all its data
VideoMetadata.objects.filter(video_id=video_id).delete()
# Automatically deletes: VideoTranscript → TranscriptSegment
```

## 📊 **AGGREGATION Patterns**

### **1. Efficient Counting and Grouping**
```python
# ✅ Count videos per search request (single query)
search_stats = SearchRequest.objects.annotate(
    video_count=Count('url_requests'),
    processed_count=Count('url_requests', filter=Q(url_requests__status='success'))
).values('request_id', 'video_count', 'processed_count')

# ✅ Get user session analytics
session_analytics = SearchSession.objects.annotate(
    search_count=Count('search_requests'),
    total_videos=Count('search_requests__url_requests'),
    total_segments=Count('search_requests__url_requests__video_metadata__video_transcript__segments')
).values('session_id', 'search_count', 'total_videos', 'total_segments')
```

## 🎯 **Search-to-Video Pipeline Efficiency**

### **Complete Pipeline Processing**
```python
def process_search_results_efficiently(search_request_id, video_urls):
    # Single query to get search context
    search_request = SearchRequest.objects.select_related('search_session').get(
        request_id=search_request_id
    )
    
    # Bulk create URL requests
    url_requests = URLRequestTable.objects.bulk_create([
        URLRequestTable(
            search_request=search_request,
            url=url,
            ip_address=search_request.search_session.user_ip
        ) for url in video_urls
    ])
    
    # Batch trigger video processing
    for url_request in url_requests:
        process_single_video.delay(url_request.request_id)
```

### **Efficient User Analytics Query**
```python
def get_user_search_history(session_id):
    # Single complex query instead of multiple simple ones
    return SearchRequest.objects.filter(
        search_session__session_id=session_id
    ).select_related('search_session').prefetch_related(
        'url_requests__video_metadata__video_transcript__segments'
    ).annotate(
        video_count=Count('url_requests'),
        segment_count=Count('url_requests__video_metadata__video_transcript__segments')
    )
```

## ⚡ **Key Performance Rules**

1. **Always use select_related** for ForeignKey/OneToOne traversal
2. **Always use prefetch_related** for reverse ForeignKey/ManyToMany  
3. **Use natural keys directly** when you have them (video_id, segment_id)
4. **Bulk operations** for creating/updating multiple records
5. **Use annotate() + values()** instead of Python loops for aggregations
6. **Leverage CASCADE deletes** instead of manual cleanup
7. **Use get_or_create** to avoid duplicate existence checks

These patterns will minimize your database round trips and optimize performance for your search-to-video processing pipeline! 🚀