from django.test import TestCase
from django.db import transaction
from api.models import URLRequestTable
from video_processor.models import VideoMetadata, VideoTranscript, TranscriptSegment


class PrimaryKeyMigrationTests(TestCase):
    """Test that the primary key migrations work correctly"""
    
    def setUp(self):
        """Set up test data"""
        self.test_video_id = "dQw4w9WgXcQ"
        self.test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        self.test_ip = "127.0.0.1"
        
        # Create URLRequestTable entry
        self.url_request = URLRequestTable.objects.create(
            url=self.test_url,
            ip_address=self.test_ip,
            status='processing'
        )
        
    def test_videometadata_uses_video_id_as_pk(self):
        """Test VideoMetadata uses video_id as primary key"""
        # Create VideoMetadata
        video_metadata = VideoMetadata.objects.create(
            url_request=self.url_request,
            video_id=self.test_video_id,
            title="Test Video",
            duration=180,
            channel_name="Test Channel",
            status='success'
        )
        
        # Check that video_id is the primary key
        self.assertEqual(video_metadata.pk, self.test_video_id)
        self.assertEqual(video_metadata.video_id, self.test_video_id)
        
        # Test retrieval by primary key
        retrieved = VideoMetadata.objects.get(pk=self.test_video_id)
        self.assertEqual(retrieved.video_id, self.test_video_id)
        self.assertEqual(retrieved.title, "Test Video")
        
    def test_videotranscript_uses_video_id_as_pk(self):
        """Test VideoTranscript uses video_id as primary key"""
        # Create VideoMetadata first
        video_metadata = VideoMetadata.objects.create(
            url_request=self.url_request,
            video_id=self.test_video_id,
            title="Test Video",
            status='success'
        )
        
        # Create VideoTranscript
        transcript = VideoTranscript.objects.create(
            video_metadata=video_metadata,
            video_id=self.test_video_id,
            transcript_text="This is a test transcript",
            status='success'
        )
        
        # Check that video_id is the primary key
        self.assertEqual(transcript.pk, self.test_video_id)
        self.assertEqual(transcript.video_id, self.test_video_id)
        
        # Test retrieval by primary key
        retrieved = VideoTranscript.objects.get(pk=self.test_video_id)
        self.assertEqual(retrieved.video_id, self.test_video_id)
        self.assertEqual(retrieved.transcript_text, "This is a test transcript")
        
    def test_transcriptsegment_uses_segment_id_as_pk(self):
        """Test TranscriptSegment uses segment_id as primary key"""
        # Create VideoMetadata and VideoTranscript first
        video_metadata = VideoMetadata.objects.create(
            url_request=self.url_request,
            video_id=self.test_video_id,
            title="Test Video",
            status='success'
        )
        
        transcript = VideoTranscript.objects.create(
            video_metadata=video_metadata,
            video_id=self.test_video_id,
            transcript_text="This is a test transcript",
            status='success'
        )
        
        # Create TranscriptSegment
        segment_id = f"{self.test_video_id}_001"
        segment = TranscriptSegment.objects.create(
            transcript=transcript,
            segment_id=segment_id,
            sequence_number=1,
            start_time=10.0,
            duration=30.0,
            text="First segment of the transcript"
        )
        
        # Check that segment_id is the primary key
        self.assertEqual(segment.pk, segment_id)
        self.assertEqual(segment.segment_id, segment_id)
        
        # Test retrieval by primary key
        retrieved = TranscriptSegment.objects.get(pk=segment_id)
        self.assertEqual(retrieved.segment_id, segment_id)
        self.assertEqual(retrieved.text, "First segment of the transcript")
        
    def test_foreign_key_relationships(self):
        """Test that foreign key relationships work correctly with natural keys"""
        # Create VideoMetadata
        video_metadata = VideoMetadata.objects.create(
            url_request=self.url_request,
            video_id=self.test_video_id,
            title="Test Video",
            status='success'
        )
        
        # Create VideoTranscript
        transcript = VideoTranscript.objects.create(
            video_metadata=video_metadata,
            video_id=self.test_video_id,
            transcript_text="This is a test transcript",
            status='success'
        )
        
        # Create TranscriptSegments
        segment1 = TranscriptSegment.objects.create(
            transcript=transcript,
            segment_id=f"{self.test_video_id}_001",
            sequence_number=1,
            start_time=10.0,
            duration=30.0,
            text="First segment"
        )
        
        segment2 = TranscriptSegment.objects.create(
            transcript=transcript,
            segment_id=f"{self.test_video_id}_002",
            sequence_number=2,
            start_time=40.0,
            duration=25.0,
            text="Second segment"
        )
        
        # Test relationships
        self.assertEqual(video_metadata.video_transcript, transcript)
        self.assertEqual(transcript.video_metadata, video_metadata)
        
        # Test reverse relationships
        segments = transcript.segments.all()
        self.assertEqual(segments.count(), 2)
        self.assertIn(segment1, segments)
        self.assertIn(segment2, segments)
        
        # Test foreign key references
        self.assertEqual(segment1.transcript, transcript)
        self.assertEqual(segment2.transcript, transcript)
        
    def test_natural_key_managers(self):
        """Test that natural key managers work correctly"""
        # Create VideoMetadata
        video_metadata = VideoMetadata.objects.create(
            url_request=self.url_request,
            video_id=self.test_video_id,
            title="Test Video",
            status='success'
        )
        
        # Test VideoMetadata natural key manager
        retrieved_metadata = VideoMetadata.objects.get_by_video_id(self.test_video_id)
        self.assertEqual(retrieved_metadata, video_metadata)
        
        # Create VideoTranscript
        transcript = VideoTranscript.objects.create(
            video_metadata=video_metadata,
            video_id=self.test_video_id,
            transcript_text="Test transcript",
            status='success'
        )
        
        # Test VideoTranscript natural key manager
        retrieved_transcript = VideoTranscript.objects.get_by_video_id(self.test_video_id)
        self.assertEqual(retrieved_transcript, transcript)
        
        # Create TranscriptSegment
        segment_id = f"{self.test_video_id}_001"
        segment = TranscriptSegment.objects.create(
            transcript=transcript,
            segment_id=segment_id,
            sequence_number=1,
            start_time=10.0,
            duration=30.0,
            text="Test segment"
        )
        
        # Test TranscriptSegment natural key manager
        retrieved_segment = TranscriptSegment.objects.get_by_segment_id(segment_id)
        self.assertEqual(retrieved_segment, segment)
        
        # Test get_by_video_id for segments
        video_segments = TranscriptSegment.objects.get_by_video_id(self.test_video_id)
        self.assertIn(segment, video_segments)
        
    def test_unique_constraints(self):
        """Test that unique constraints are properly enforced"""
        # Create first VideoMetadata
        video_metadata1 = VideoMetadata.objects.create(
            url_request=self.url_request,
            video_id=self.test_video_id,
            title="Test Video 1",
            status='success'
        )
        
        # Create second URLRequestTable for second video
        url_request2 = URLRequestTable.objects.create(
            url="https://www.youtube.com/watch?v=test2",
            ip_address=self.test_ip,
            status='processing'
        )
        
        # Try to create another VideoMetadata with same video_id - should fail
        with self.assertRaises(Exception):
            VideoMetadata.objects.create(
                url_request=url_request2,
                video_id=self.test_video_id,  # Same video_id
                title="Test Video 2",
                status='success'
            )
            
    def test_cascade_deletion(self):
        """Test that cascade deletion works correctly"""
        # Create full hierarchy
        video_metadata = VideoMetadata.objects.create(
            url_request=self.url_request,
            video_id=self.test_video_id,
            title="Test Video",
            status='success'
        )
        
        transcript = VideoTranscript.objects.create(
            video_metadata=video_metadata,
            video_id=self.test_video_id,
            transcript_text="Test transcript",
            status='success'
        )
        
        segment = TranscriptSegment.objects.create(
            transcript=transcript,
            segment_id=f"{self.test_video_id}_001",
            sequence_number=1,
            start_time=10.0,
            duration=30.0,
            text="Test segment"
        )
        
        # Delete VideoMetadata - should cascade to VideoTranscript and TranscriptSegment
        video_metadata.delete()
        
        # Check that related objects are deleted
        self.assertFalse(VideoMetadata.objects.filter(video_id=self.test_video_id).exists())
        self.assertFalse(VideoTranscript.objects.filter(video_id=self.test_video_id).exists())
        self.assertFalse(TranscriptSegment.objects.filter(segment_id=f"{self.test_video_id}_001").exists())