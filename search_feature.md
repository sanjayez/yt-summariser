# **YouTube Topic Search Feature - Implementation Guide**

## **📋 Project Summary**

### **Requirements**
- Create `/api/topic/search/` endpoint accepting natural language queries
- Implement session-based user tracking with IP address for rate limiting
- Use LLM to enhance user queries for better YouTube search results
- Return 5 YouTube video URLs from search results
- Maintain dependency-agnostic search layer for future flexibility

### **Expected Deliverables**
- New `topic` Django app with clean separation of concerns
- Session management system (hybrid session ID + IP tracking)
- LLM-powered query enhancement using existing ai_utils
- YouTube search functionality using scrapetube
- Structured JSON API response with search metadata

### **Success Criteria**
- ✅ Endpoint responds with video URLs for any natural language query
- ✅ Session tracking works across multiple requests
- ✅ LLM improves search query effectiveness
- ✅ Database properly stores search history
- ✅ Clean, maintainable code architecture

---

## **🏗️ Implementation Strategy**

### **Phase 1: Foundation Setup**
Set up the basic Django app structure and dependencies

### **Phase 2: Data Layer**
Create database models for session and search tracking

### **Phase 3: Service Layer**
Build abstracted search and query processing services

### **Phase 4: API Layer**
Implement the main endpoint with all integrations

### **Phase 5: Testing & Validation**
Ensure everything works end-to-end

---

## **✅ Implementation Checklist**

### **Phase 1: Foundation Setup**
- [ ] **Install Dependencies**
  ```bash
  pip install scrapetube
  ```
- [ ] **Create Topic App**
  ```bash
  python manage.py startapp topic
  ```
- [ ] **Add to Django Settings**
  - [ ] Add `'topic'` to `INSTALLED_APPS` in `settings.py`
- [ ] **Set Up URL Routing**
  - [ ] Add `path('api/topic/', include('topic.urls'))` to main `urls.py`
  - [ ] Create `topic/urls.py` file

### **Phase 2: Data Layer**
- [ ] **Create Database Models** (`topic/models.py`)
  - [ ] `SearchSession` model with session_id, ip_address, timestamps
  - [ ] `SearchRequest` model with request_id, session FK, queries, status
- [ ] **Create and Run Migrations**
  ```bash
  python manage.py makemigrations topic
  python manage.py migrate
  ```
- [ ] **Test Database Models**
  - [ ] Verify tables created correctly
  - [ ] Test model creation in Django shell

### **Phase 3: Service Layer**
- [ ] **Create Services Directory Structure**
  ```
  topic/services/
  ├── __init__.py
  ├── search_service.py
  ├── query_processor.py
  └── providers/
      ├── __init__.py
      └── scrapetube_provider.py
  ```
- [ ] **Implement Search Abstraction** (`topic/services/search_service.py`)
  - [ ] Create `SearchProvider` ABC
  - [ ] Create `YouTubeSearchService` wrapper class
- [ ] **Implement Scrapetube Provider** (`topic/services/providers/scrapetube_provider.py`)
  - [ ] Create `ScrapeTubeProvider` class
  - [ ] Implement search method returning YouTube URLs
  - [ ] Test with sample queries
- [ ] **Implement Query Processor** (`topic/services/query_processor.py`)
  - [ ] Create `QueryProcessor` class
  - [ ] Integrate with existing ai_utils LLMService
  - [ ] Create effective prompt for query enhancement
  - [ ] Test LLM query improvements

### **Phase 4: API Layer**
- [ ] **Create Utility Functions** (`topic/utils/`)
  - [ ] Create `session_utils.py`
  - [ ] Implement `get_or_create_session()` function
  - [ ] Reuse `get_client_ip()` from api.utils
- [ ] **Implement Main Endpoint** (`topic/views.py`)
  - [ ] Create `search_topic` view function
  - [ ] Add session management logic
  - [ ] Integrate LLM query processing
  - [ ] Integrate YouTube search
  - [ ] Create database records
  - [ ] Structure JSON response
- [ ] **Configure URL Patterns** (`topic/urls.py`)
  - [ ] Add search endpoint route
  - [ ] Test URL routing works

### **Phase 5: Testing & Validation**
- [ ] **Unit Testing**
  - [ ] Test `ScrapeTubeProvider.search()` method
  - [ ] Test `QueryProcessor.enhance_query()` method
  - [ ] Test session creation and retrieval
- [ ] **Integration Testing**
  - [ ] Test full endpoint with curl/Postman
  - [ ] Verify database records creation
  - [ ] Test with different query types
  - [ ] Test session persistence across requests
- [ ] **Error Handling**
  - [ ] Handle scrapetube failures gracefully
  - [ ] Handle LLM service errors
  - [ ] Add appropriate HTTP status codes
  - [ ] Add input validation

### **Phase 6: Documentation & Cleanup**
- [ ] **Code Documentation**
  - [ ] Add docstrings to all classes and methods
  - [ ] Add inline comments for complex logic
- [ ] **API Documentation**
  - [ ] Document endpoint request/response format
  - [ ] Add example requests and responses
- [ ] **Code Review**
  - [ ] Ensure consistent code style
  - [ ] Check for proper error handling
  - [ ] Verify separation of concerns

---

## **📁 File Structure Reference**

```
topic/
├── __init__.py
├── admin.py
├── apps.py
├── models.py                    # SearchSession, SearchRequest
├── views.py                     # search_topic endpoint
├── urls.py                      # URL patterns
├── migrations/
├── services/
│   ├── __init__.py
│   ├── search_service.py        # Search abstraction layer
│   ├── query_processor.py       # LLM query enhancement
│   └── providers/
│       ├── __init__.py
│       └── scrapetube_provider.py  # YouTube search implementation
├── utils/
│   ├── __init__.py
│   └── session_utils.py         # Session management utilities
└── tests/
    ├── __init__.py
    ├── test_models.py
    ├── test_views.py
    └── test_services.py
```

---

## **🧪 Testing Commands**

### **Test Endpoint**
```bash
curl -X POST http://localhost:8000/api/topic/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "how to learn machine learning for beginners"}'
```

### **Expected Response**
```json
{
  "search_request_id": "uuid-here",
  "session_id": "uuid-here", 
  "original_query": "how to learn machine learning for beginners",
  "processed_query": "machine learning tutorial beginner guide",
  "video_urls": ["https://youtube.com/watch?v=...", ...],
  "total_videos": 5
}
```

---

## **🚀 Next Steps (Future Phases)**
- [ ] **Rate Limiting Implementation**
- [ ] **Video Processing Pipeline Integration** 
- [ ] **Server-Sent Events for Real-time Updates**
- [ ] **Result Aggregation System**
- [ ] **Caching Layer for Search Results**

---

## **⚠️ Notes & Considerations**
- Keep scrapetube as dependency-agnostic as possible for easy swapping
- Monitor scrapetube for YouTube DOM changes that might break functionality
- Consider implementing fallback search providers for reliability
- Test thoroughly with various query types and edge cases