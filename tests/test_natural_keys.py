"""
Test natural key functionality for the database normalization changes.
"""
from django.test import TestCase
from django.db import transaction
from api.models import URLRequestTable
from video_processor.models import VideoMetadata, VideoTranscript, TranscriptSegment
from topic.models import SearchSession, SearchRequest


class NaturalKeyTestCase(TestCase):
    """Test natural key functionality and relationships"""
    
    def setUp(self):
        """Set up test data"""
        # Create search session and request
        self.session = SearchSession.objects.create(
            user_ip='192.168.1.1',
            status='processing'
        )
        
        self.search_request = SearchRequest.objects.create(
            search_session=self.session,
            original_query='python tutorial',
            video_urls=['https://youtube.com/watch?v=dQw4w9WgXcQ'],
            status='success'
        )
        
        # Create URL request linked to search
        self.url_request = URLRequestTable.objects.create(
            search_request=self.search_request,
            url='https://youtube.com/watch?v=dQw4w9WgXcQ',
            ip_address='192.168.1.1',
            status='processing'
        )
        
        # Create video metadata
        self.video_metadata = VideoMetadata.objects.create(
            video_id='dQw4w9WgXcQ',
            url_request=self.url_request,
            title='Test Video',
            status='success'
        )
        
        # Create video transcript
        self.video_transcript = VideoTranscript.objects.create(
            video_id='dQw4w9WgXcQ',
            video_metadata=self.video_metadata,  # Required foreign key field
            transcript_text='Test transcript',
            status='success'
        )
        
        # Create transcript segments
        self.segment1 = TranscriptSegment.objects.create(
            transcript=self.video_transcript,
            segment_id='dQw4w9WgXcQ_001',
            sequence_number=1,
            start_time=0.0,
            duration=10.0,
            text='First segment'
        )
        
        self.segment2 = TranscriptSegment.objects.create(
            transcript=self.video_transcript,
            segment_id='dQw4w9WgXcQ_002',
            sequence_number=2,
            start_time=10.0,
            duration=10.0,
            text='Second segment'
        )
    
    def test_search_to_video_pipeline(self):
        """Test complete flow from search to video processing"""
        # Verify relationships
        self.assertEqual(self.url_request.search_request, self.search_request)
        self.assertEqual(self.video_metadata.url_request, self.url_request)
        self.assertEqual(self.video_transcript.video_metadata, self.video_metadata)
        self.assertEqual(self.segment1.transcript, self.video_transcript)
        
        # Verify reverse relationships
        self.assertEqual(self.search_request.url_requests.first(), self.url_request)
        self.assertEqual(self.video_metadata.video_transcript, self.video_transcript)
        self.assertEqual(self.video_transcript.segments.count(), 2)
    
    def test_natural_key_lookups(self):
        """Test natural key lookup methods"""
        # Test VideoMetadata natural key lookup
        video = VideoMetadata.objects.get_by_video_id('dQw4w9WgXcQ')
        self.assertEqual(video, self.video_metadata)
        self.assertEqual(video.natural_key(), ('dQw4w9WgXcQ',))
        
        # Test VideoTranscript natural key lookup
        transcript = VideoTranscript.objects.get_by_video_id('dQw4w9WgXcQ')
        self.assertEqual(transcript, self.video_transcript)
        self.assertEqual(transcript.natural_key(), ('dQw4w9WgXcQ',))
        
        # Test TranscriptSegment natural key lookup
        segment = TranscriptSegment.objects.get_by_segment_id('dQw4w9WgXcQ_001')
        self.assertEqual(segment, self.segment1)
        self.assertEqual(segment.natural_key(), ('dQw4w9WgXcQ_001',))
        
        # Test segment video_id extraction
        self.assertEqual(segment.get_video_id(), 'dQw4w9WgXcQ')
    
    def test_segment_video_id_filtering(self):
        """Test filtering segments by video_id"""
        segments = TranscriptSegment.objects.get_by_video_id('dQw4w9WgXcQ')
        self.assertEqual(segments.count(), 2)
        self.assertIn(self.segment1, segments)
        self.assertIn(self.segment2, segments)
    
    def test_aggregation_queries(self):
        """Test aggregating videos by search request"""
        # Test video count for search request
        video_count = self.search_request.url_requests.count()
        self.assertEqual(video_count, 1)
        
        # Test status aggregation
        from django.db.models import Count
        status_counts = self.search_request.url_requests.values('status').annotate(
            count=Count('status')
        )
        self.assertEqual(status_counts[0]['count'], 1)
        self.assertEqual(status_counts[0]['status'], 'processing')
    
    def test_pinecone_vector_id(self):
        """Test that segment_id can be used as Pinecone vector ID"""
        # Verify segment_id is suitable for vector store
        self.assertEqual(self.segment1.pinecone_vector_id, 'dQw4w9WgXcQ_001')
        self.assertEqual(self.segment2.pinecone_vector_id, 'dQw4w9WgXcQ_002')
        
        # Verify uniqueness constraint
        with self.assertRaises(Exception):
            TranscriptSegment.objects.create(
                transcript=self.video_transcript,
                segment_id='dQw4w9WgXcQ_001',  # Duplicate segment_id
                sequence_number=3,
                start_time=20.0,
                duration=10.0,
                text='Duplicate segment'
            )
    
    def test_youtube_url_generation(self):
        """Test YouTube URL generation with timestamps"""
        url = self.segment1.get_youtube_url_with_timestamp()
        self.assertEqual(url, 'https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=0s')
        
        url = self.segment2.get_youtube_url_with_timestamp()
        self.assertEqual(url, 'https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s')
    
    def test_video_metadata_properties(self):
        """Test VideoMetadata property methods"""
        self.assertEqual(self.video_metadata.webpage_url, 'https://www.youtube.com/watch?v=dQw4w9WgXcQ')
        self.assertEqual(str(self.video_metadata), 'dQw4w9WgXcQ - Test Video')
    
    def test_video_transcript_properties(self):
        """Test VideoTranscript property methods"""
        self.assertEqual(str(self.video_transcript), 'Transcript for dQw4w9WgXcQ')
        
        # Test formatted transcript
        formatted = self.video_transcript.get_formatted_transcript()
        self.assertEqual(len(formatted), 2)
        self.assertEqual(formatted[0]['timestamp'], '00:00')
        self.assertEqual(formatted[1]['timestamp'], '00:10')