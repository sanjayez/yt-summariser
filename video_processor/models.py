from django.db import models

from api.models import URLRequestTable


class VideoMetadataManager(models.Manager):
    """Custom manager for VideoMetadata with natural key support"""

    def get_by_video_id(self, video_id):
        """Get VideoMetadata by natural key (video_id)"""
        return self.get(video_id=video_id)

    def get_or_create_by_video_id(self, video_id, defaults=None):
        """Get or create VideoMetadata by natural key (video_id)"""
        return self.get_or_create(video_id=video_id, defaults=defaults or {})


class VideoTranscriptManager(models.Manager):
    """Custom manager for VideoTranscript with natural key support"""

    def get_by_video_id(self, video_id):
        """Get VideoTranscript by natural key (video_id)"""
        return self.get(video_id=video_id)

    def get_or_create_by_video_id(self, video_id, defaults=None):
        """Get or create VideoTranscript by natural key (video_id)"""
        return self.get_or_create(video_id=video_id, defaults=defaults or {})


class TranscriptSegmentManager(models.Manager):
    """Custom manager for TranscriptSegment with natural key support"""

    def get_by_segment_id(self, segment_id):
        """Get TranscriptSegment by natural key (segment_id)"""
        return self.get(segment_id=segment_id)

    def get_or_create_by_segment_id(self, segment_id, defaults=None):
        """Get or create TranscriptSegment by natural key (segment_id)"""
        return self.get_or_create(segment_id=segment_id, defaults=defaults or {})

    def get_by_video_id(self, video_id):
        """Get all segments for a video by video_id"""
        return self.filter(segment_id__startswith=f"{video_id}_")


class VideoMetadata(models.Model):
    STATUS_CHOICES = [
        ("processing", "Processing"),
        ("failed", "Failed"),
        ("success", "Success"),
    ]

    objects = VideoMetadataManager()  # Custom manager

    url_request = models.OneToOneField(
        URLRequestTable, on_delete=models.CASCADE, related_name="video_metadata"
    )
    video_id = models.CharField(
        max_length=20,
        primary_key=True,
        help_text="YouTube video ID (e.g., 'dQw4w9WgXcQ')",
    )
    title = models.CharField(max_length=500, blank=True)
    description = models.TextField(blank=True)
    duration = models.IntegerField(null=True, blank=True)  # in seconds
    channel_name = models.CharField(max_length=200, blank=True)
    view_count = models.BigIntegerField(null=True, blank=True)

    # New fields from YouTube API
    upload_date = models.DateField(
        null=True, blank=True, help_text="When video was published"
    )
    language = models.CharField(
        max_length=12,
        null=True,
        blank=True,
        default="en",
        help_text="Primary language of the video",
    )
    like_count = models.BigIntegerField(
        null=True, blank=True, help_text="Number of likes"
    )
    comment_count = models.BigIntegerField(
        null=True, blank=True, help_text="Number of comments"
    )
    channel_id = models.CharField(
        max_length=50, blank=True, help_text="YouTube channel ID"
    )
    channel_thumbnail = models.URLField(
        blank=True, help_text="Channel avatar/thumbnail URL"
    )
    tags = models.JSONField(
        default=list, blank=True, help_text="Video tags for categorization"
    )
    categories = models.JSONField(
        default=list, blank=True, help_text="YouTube categories"
    )
    thumbnail = models.URLField(blank=True, help_text="Video thumbnail URL")
    channel_follower_count = models.BigIntegerField(
        null=True, blank=True, help_text="Channel subscriber count"
    )
    channel_is_verified = models.BooleanField(
        default=False, help_text="Whether channel is verified"
    )
    uploader_id = models.CharField(
        max_length=100, blank=True, help_text="Channel handle (e.g., '@TEDx')"
    )

    # Vector embedding tracking
    is_embedded = models.BooleanField(
        default=False,
        help_text="Whether video metadata has been embedded in vector store",
    )
    engagement = models.JSONField(
        default=list,
        blank=True,
        help_text="High engagement segments (>95% from heatmap)",
    )

    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default="processing"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    @property
    def duration_string(self):
        """Return human-readable duration (MM:SS format)"""
        if not self.duration:
            return "0:00"
        minutes = self.duration // 60
        seconds = self.duration % 60
        return f"{minutes}:{seconds:02d}"

    @property
    def webpage_url(self):
        """Return direct YouTube URL"""
        if self.video_id:
            return f"https://www.youtube.com/watch?v={self.video_id}"
        return ""

    @property
    def channel_thumbnail_url(self):
        """Alias for channel thumbnail used consistently across code paths"""
        return self.channel_thumbnail

    def natural_key(self):
        """Return natural key for this model"""
        return (self.video_id,)

    def __str__(self):
        return f"{self.video_id} - {self.title[:50]}"

    class Meta:
        indexes = [
            models.Index(fields=["status"]),
            models.Index(fields=["created_at"]),
            models.Index(fields=["channel_id"]),  # For channel-based filtering
            models.Index(fields=["language"]),  # For language-based filtering
            models.Index(fields=["is_embedded"]),  # For vector embedding tracking
            models.Index(fields=["comment_count"]),  # For engagement-based queries
        ]


class VideoTranscript(models.Model):
    STATUS_CHOICES = [
        ("processing", "Processing"),
        ("failed", "Failed"),
        ("success", "Success"),
    ]

    objects = VideoTranscriptManager()  # Custom manager

    # Natural key field for efficient lookups and primary key
    video_id = models.CharField(
        max_length=20,
        primary_key=True,
        help_text="YouTube video ID - natural key for lookups",
    )

    # Proper foreign key relationship to VideoMetadata for cascading deletes
    video_metadata = models.OneToOneField(
        VideoMetadata,
        on_delete=models.CASCADE,
        to_field="video_id",  # Reference the natural key
        related_name="video_transcript",
        help_text="Foreign key to VideoMetadata for proper cascading deletes",
    )

    transcript_text = (
        models.TextField()
    )  # Plain text for search/summary - only field needed for summaries

    # AI-generated summary fields
    summary = models.TextField(
        blank=True, null=True, help_text="AI-generated summary of the video content"
    )
    key_points = models.JSONField(
        default=list,
        blank=True,
        help_text="List of key points extracted from the video",
    )
    chapters = models.JSONField(
        default=list,
        blank=True,
        help_text="List of chapters with one-line summaries: [{'chapter': int, 'title': str, 'summary': str}]",
    )

    # Transcript source tracking
    TRANSCRIPT_SOURCES = [
        ("decodo", "Decodo API"),
        ("youtube_api", "YouTube Transcript API"),
        ("manual", "Manual Upload"),
    ]

    transcript_source = models.CharField(
        max_length=20,
        choices=TRANSCRIPT_SOURCES,
        default="decodo",
        help_text="Source used for transcript extraction",
    )

    # Content analysis fields have been moved to ContentAnalysis model

    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default="processing"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def get_formatted_transcript(self):
        """Return transcript with clickable timestamps for UI using segments"""
        if not hasattr(self, "segments") or not self.segments.exists():
            # If no segments, return a single segment with the full text
            return [
                {"timestamp": "00:00", "start_seconds": 0, "text": self.transcript_text}
            ]

        formatted_segments = []
        for segment in self.segments.all():
            # Format timestamp as MM:SS
            minutes = int(segment.start_time // 60)
            seconds = int(segment.start_time % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"
            formatted_segments.append(
                {
                    "timestamp": timestamp,
                    "start_seconds": segment.start_time,
                    "text": segment.text,
                }
            )
        return formatted_segments

    def natural_key(self):
        """Return natural key for this model"""
        return (self.video_id,)

    def __str__(self):
        return f"Transcript for {self.video_id}"

    class Meta:
        indexes = [
            models.Index(fields=["status"]),
            models.Index(fields=["created_at"]),
        ]


class TranscriptSegment(models.Model):
    """
    Individual transcript segments for vector embedding and precise video navigation.
    Each segment represents a small portion of the video with timestamps.
    """

    objects = TranscriptSegmentManager()  # Custom manager

    transcript = models.ForeignKey(
        VideoTranscript,
        on_delete=models.CASCADE,
        related_name="segments",
        to_field="video_id",
    )
    segment_id = models.CharField(
        max_length=50,
        primary_key=True,
        help_text="Unique segment ID format: 'video_id_segment_number' (e.g., 'dQw4w9WgXcQ_001')",
    )
    sequence_number = models.IntegerField(
        help_text="Sequential order of this segment in the video"
    )

    # Timestamp information
    start_time = models.FloatField(help_text="Start time in seconds")
    duration = models.FloatField(help_text="Duration of this segment in seconds")

    # Content
    text = models.TextField(help_text="Text content of this segment")

    # Vector embedding tracking
    is_embedded = models.BooleanField(
        default=False,
        help_text="Whether this segment has been embedded in vector store",
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["sequence_number"]
        unique_together = ["transcript", "sequence_number"]
        indexes = [
            models.Index(fields=["transcript", "sequence_number"]),
            models.Index(fields=["start_time"]),
            models.Index(fields=["is_embedded"]),
        ]

    @property
    def end_time(self):
        """Calculate end time from next segment or use start_time + duration"""
        next_segment = self.transcript.segments.filter(
            sequence_number=self.sequence_number + 1
        ).first()
        if next_segment:
            return next_segment.start_time
        return self.start_time + self.duration

    @property
    def pinecone_vector_id(self):
        """Use segment_id as the Pinecone vector ID"""
        return self.segment_id

    def __str__(self):
        return f"{self.segment_id} ({self.start_time}s-{self.end_time}s)"

    def get_formatted_timestamp(self):
        """Return formatted timestamp as MM:SS for UI display"""
        minutes = int(self.start_time // 60)
        seconds = int(self.start_time % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def get_youtube_url_with_timestamp(self):
        """Return YouTube URL with timestamp for direct video navigation"""
        video_id = self.transcript.video_metadata.video_id
        return f"https://www.youtube.com/watch?v={video_id}&t={int(self.start_time)}s"

    def natural_key(self):
        """Return natural key for this model"""
        return (self.segment_id,)

    def get_video_id(self):
        """Extract video_id from segment_id"""
        return self.segment_id.rsplit("_", 1)[0] if "_" in self.segment_id else None


# Helper function to update URLRequestTable status based on related models
def update_url_request_status(url_request):
    """Update URLRequestTable status based on VideoMetadata and VideoTranscript statuses"""
    try:
        metadata_status = getattr(url_request.video_metadata, "status", None)
        # Get transcript status through the video_transcript property
        transcript_status = None
        if hasattr(url_request, "video_metadata") and url_request.video_metadata:
            video_transcript = url_request.video_metadata.video_transcript
            if video_transcript:
                transcript_status = video_transcript.status

        # If both exist and both are successful (handle both 'success' and 'completed' as success)
        metadata_success = metadata_status in ["success", "completed"]
        transcript_success = transcript_status == "success"

        # Check if video is excluded - this takes precedence over metadata/transcript status
        from video_processor.utils.video_filtering import extract_video_id_from_url

        video_id = extract_video_id_from_url(url_request.url)
        is_excluded = video_id and VideoExclusionTable.is_excluded(video_id)

        if is_excluded:
            # Video was excluded by classifier - keep as failed
            url_request.status = "failed"
            # Set failure reason if not already set
            if not url_request.failure_reason:
                url_request.failure_reason = "excluded"
        elif metadata_success and transcript_success:
            url_request.status = "success"
            # Clear failure reason on success
            url_request.failure_reason = None
        # If either has failed
        elif metadata_status == "failed" or transcript_status == "failed":
            url_request.status = "failed"
            # ONLY set failure reason if not already set by processors
            # Processors know the specific failure type better than this general function
            if not url_request.failure_reason:
                # Determine failure reason based on which component failed
                # But prefer transcript failure over metadata failure for more specific diagnosis
                if transcript_status == "failed":
                    url_request.failure_reason = "no_transcript"
                elif metadata_status == "failed":
                    url_request.failure_reason = "no_metadata"
        # Otherwise keep as processing
        else:
            url_request.status = "processing"

        url_request.save()

    except (VideoMetadata.DoesNotExist, VideoTranscript.DoesNotExist):
        # If either model doesn't exist yet, keep as processing
        url_request.status = "processing"
        url_request.save()


class VideoExclusionTable(models.Model):
    """
    Tracks videos that cannot be processed for business reasons.

    This table serves as:
    1. A lookup table to prevent reprocessing known problematic videos
    2. Analytics source for understanding video suitability patterns
    3. Business intelligence for platform optimization
    """

    EXCLUSION_REASONS = [
        ("privacy_restricted", "Privacy Restricted"),
        ("language_unsupported", "Language Not Supported"),
        ("content_unavailable", "Content Unavailable"),
        ("duration_too_short", "Duration Too Short"),
        ("duration_too_long", "Duration Too Long"),
        ("transcript_unavailable", "No Transcript Available"),
        (
            "background_music_only",
            "Background Music Only",
        ),  # Videos with background music that interfere with Q&A
    ]

    video_id = models.CharField(
        max_length=20, unique=True, help_text="YouTube video ID (extracted from URL)"
    )
    video_url = models.URLField(max_length=500, help_text="Original YouTube video URL")
    exclusion_reason = models.CharField(
        max_length=30,
        choices=EXCLUSION_REASONS,
        help_text="Business reason why this video cannot be processed",
    )
    detected_at = models.DateTimeField(
        auto_now_add=True, help_text="When this exclusion was first detected"
    )

    class Meta:
        ordering = ["-detected_at"]
        indexes = [
            models.Index(fields=["video_id"]),  # Fast lookup during pre-filtering
            models.Index(fields=["exclusion_reason"]),  # Analytics queries
            models.Index(fields=["detected_at"]),  # Time-based analytics
        ]

    def __str__(self):
        return f"Excluded: {self.video_id} ({self.get_exclusion_reason_display()})"

    @classmethod
    def is_excluded(cls, video_id: str) -> bool:
        """
        Quick check if a video is in the exclusion list.

        Args:
            video_id: YouTube video ID to check

        Returns:
            bool: True if video should be excluded from processing
        """
        return cls.objects.filter(video_id=video_id).exists()

    @classmethod
    def get_exclusion_reason(cls, video_id: str) -> str:
        """
        Get the exclusion reason for a video.

        Args:
            video_id: YouTube video ID to check

        Returns:
            str: Exclusion reason or None if not excluded
        """
        try:
            return cls.objects.get(video_id=video_id).exclusion_reason
        except cls.DoesNotExist:
            return None


class ContentAnalysis(models.Model):
    """
    Two-phase content analysis for video transcripts.

    Phase 1: Preliminary analysis without timestamps (runs in parallel with summary/classification)
    Phase 2: Final analysis with timestamps and ratios (runs after embedding)
    """

    # Core relationship
    video_transcript = models.OneToOneField(
        VideoTranscript, on_delete=models.CASCADE, related_name="content_analysis"
    )

    # Phase 1: Preliminary Analysis (no timestamps needed)
    preliminary_analysis_status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending"),
            ("processing", "Processing"),
            ("completed", "Completed"),
            ("failed", "Failed"),
        ],
        default="pending",
        help_text="Status of preliminary content analysis (ads/filler detection, tone analysis)",
    )

    # Content classification results from Phase 1 (NO ratios yet - require timestamps)
    speaker_tones = models.JSONField(
        default=list,
        blank=True,
        help_text="List of detected speaker tones: ['informal', 'positive', 'humorous']",
    )

    # Raw analysis results from Phase 1 (NO timestamps)
    raw_ad_segments = models.JSONField(
        default=list,
        blank=True,
        help_text='Raw ad segments with text excerpts only: [{"text": "...", "description": "..."}]',
    )

    raw_filler_segments = models.JSONField(
        default=list,
        blank=True,
        help_text='Raw filler segments with text excerpts only: [{"text": "...", "description": "..."}]',
    )

    # Phase 2: Timestamped Analysis (after embedding)
    timestamped_analysis_status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending"),
            ("processing", "Processing"),
            ("completed", "Completed"),
            ("failed", "Failed"),
        ],
        default="pending",
        help_text="Status of final timestamped analysis and ratio calculation",
    )

    # Final timestamped results & quality ratios (REQUIRES timestamps from Phase 2)
    content_rating = models.FloatField(
        null=True,
        blank=True,
        help_text="0.0-1.0 content quality rating (1 - ad_ratio - filler_ratio)",
    )

    ad_duration_ratio = models.FloatField(
        null=True, blank=True, help_text="Total ad duration / video duration"
    )

    filler_duration_ratio = models.FloatField(
        null=True, blank=True, help_text="Total filler duration / video duration"
    )

    ad_segments = models.JSONField(
        default=list,
        blank=True,
        help_text='Final ad segments with timestamps: [{"start": 30, "end": 60, "desc": "..."}]',
    )

    filler_segments = models.JSONField(
        default=list,
        blank=True,
        help_text='Final filler segments with timestamps: [{"start": 0, "end": 15, "desc": "..."}]',
    )

    content_segments = models.JSONField(
        default=list,
        blank=True,
        help_text='Final content segments with timestamps: [{"start": 15, "end": 30, "desc": "..."}]',
    )

    # Metadata
    preliminary_completed_at = models.DateTimeField(
        null=True, blank=True, help_text="When preliminary analysis was completed"
    )

    final_completed_at = models.DateTimeField(
        null=True, blank=True, help_text="When final timestamped analysis was completed"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["preliminary_analysis_status"]),
            models.Index(fields=["timestamped_analysis_status"]),
            models.Index(fields=["content_rating"]),
            models.Index(fields=["created_at"]),
        ]

    def __str__(self):
        return f"ContentAnalysis for {self.video_transcript.video_id} (P1:{self.preliminary_analysis_status}, P2:{self.timestamped_analysis_status})"

    @property
    def is_preliminary_complete(self):
        """Check if preliminary analysis is complete"""
        return self.preliminary_analysis_status == "completed"

    @property
    def is_final_complete(self):
        """Check if final timestamped analysis is complete"""
        return self.timestamped_analysis_status == "completed"

    @property
    def is_complete(self):
        """Check if both phases are complete"""
        return self.is_preliminary_complete and self.is_final_complete

    def get_analysis_summary(self):
        """Get a summary of the content analysis results"""
        if not self.is_complete:
            return {
                "status": "incomplete",
                "preliminary_complete": self.is_preliminary_complete,
                "final_complete": self.is_final_complete,
            }

        return {
            "status": "complete",
            "content_rating": self.content_rating,
            "ad_ratio": self.ad_duration_ratio,
            "filler_ratio": self.filler_duration_ratio,
            "speaker_tones": self.speaker_tones,
            "total_ad_segments": len(self.ad_segments),
            "total_filler_segments": len(self.filler_segments),
        }
