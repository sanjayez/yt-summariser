Principle: Choose the Right Identifier for Each Purpose
Natural Keys (when available)
UUIDs (when external/global reference needed)
Auto-increment (only for pure internal, never-exposed relationships)

SearchSession (1) ──→ (M) SearchRequest (1) ──→ (M) URLRequestTable (1) ──→ (1) VideoMetadata (1) ──→ (1) VideoTranscript (1) ──→ (M) TranscriptSegment

Step 1: DB migrations, new links
Step 2: update tasks and endpoints for topic discovery and single video processing
step 3: test topic discovery and single video for consistency
step 4: Improve result aggregation and parallel processing with benchmarks and resiliency


Current status - 10 - Jul 8:30 AM

☐ Phase 4: Clean up redundant fields and optimize indexes
     ☐ Phase 3: Switch VideoMetadata to use video_id as primary key
     ☐ Phase 3: Switch TranscriptSegment to use segment_id as primary key
     ☐ Phase 3: Update VideoTranscript to use video_id FK

⏺ Database Normalization Implementation Summary

  I have successfully implemented the database normalization changes from your plan. Here's what was accomplished:

  ✅ Completed - High Priority Items:

  1. ✅ Phase 1: Search-to-Video Pipeline
    - Added search_request foreign key to URLRequestTable
    - Created proper relationship linking search requests to video processing
    - Added database indexes for efficient lookups
  2. ✅ Phase 2: Video ID Normalization
    - Created data migration to populate missing video_id values
    - Made video_id field non-nullable with proper validation
    - All existing records properly migrated
  3. ✅ Enhanced Model Structure
    - Added custom managers with natural key lookup methods
    - Added video_id field to VideoTranscript for efficient lookups
    - Created proper indexes for performance optimization
    - Added natural key methods to all models
  4. ✅ New API Endpoint
    - Created /api/topic/search-and-process/ endpoint that combines search and video processing
    - Properly populates the search_request relationship when videos are processed
    - Enables complete search-to-video tracking pipeline
  5. ✅ Comprehensive Testing
    - All existing tests continue to pass
    - Created comprehensive test suite for natural key functionality
    - Validated search-to-video pipeline relationships
    - Tested aggregation queries and performance optimizations

  🔄 Deferred - Lower Priority Items:

  The following complex primary key migrations were deferred due to SQLite foreign key constraint complexity:
  - Switching VideoMetadata to use video_id as primary key
  - Switching TranscriptSegment to use segment_id as primary key
  - Complex foreign key relationship updates

  However, the functional benefits are achieved through:
  - Natural key lookup methods (get_by_video_id, get_by_segment_id)
  - Proper indexing for efficient queries
  - Clean API for natural key operations
  - All the search-to-video tracking functionality

  🎯 Key Benefits Achieved:

  1. Clear Data Lineage: SearchSession → SearchRequest → URLRequestTable → VideoMetadata → VideoTranscript → 
  TranscriptSegment
  2. Efficient Natural Key Lookups:
  # Easy video lookups by YouTube ID
  video = VideoMetadata.objects.get_by_video_id('dQw4w9WgXcQ')
  transcript = VideoTranscript.objects.get_by_video_id('dQw4w9WgXcQ')
  segment = TranscriptSegment.objects.get_by_segment_id('dQw4w9WgXcQ_001')
  3. Search-to-Video Integration: The new /api/topic/search-and-process/ endpoint enables full workflow integration
  4. Vector Store Compatibility: segment_id works perfectly as Pinecone vector ID
  5. Robust Testing: Comprehensive test coverage ensures reliability


