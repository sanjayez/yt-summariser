from django.core.management.base import BaseCommand
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
from video_processor.models import VideoTranscript
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Reprocess existing transcripts to add timestamp data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--transcript-id',
            type=int,
            help='Reprocess specific transcript by ID',
        )
        parser.add_argument(
            '--all',
            action='store_true',
            help='Reprocess all transcripts missing timestamp data',
        )

    def handle(self, *args, **options):
        if options['transcript_id']:
            transcripts = VideoTranscript.objects.filter(id=options['transcript_id'])
        elif options['all']:
            transcripts = VideoTranscript.objects.filter(
                transcript_data__isnull=True,
                status='success'
            )
        else:
            self.stdout.write(
                self.style.ERROR('Please specify either --transcript-id or --all')
            )
            return

        self.stdout.write(f'Found {transcripts.count()} transcripts to reprocess')

        for transcript in transcripts:
            try:
                # Extract video ID from URL
                url = transcript.url_request.url
                video_id = self.extract_video_id(url)
                
                if not video_id:
                    self.stdout.write(
                        self.style.ERROR(f'Could not extract video ID from {url}')
                    )
                    continue

                self.stdout.write(f'Reprocessing transcript for video {video_id}...')
                
                # Get transcript with timestamps from YouTube
                youtube_transcript = YouTubeTranscriptApi.get_transcript(video_id)
                
                # Update the transcript record
                transcript.transcript_data = youtube_transcript
                transcript.save()
                
                self.stdout.write(
                    self.style.SUCCESS(
                        f'Successfully updated transcript {transcript.id} with {len(youtube_transcript)} segments'
                    )
                )

            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(
                        f'Failed to reprocess transcript {transcript.id}: {str(e)}'
                    )
                )

    def extract_video_id(self, url):
        """Extract video ID from YouTube URL"""
        try:
            ydl_opts = {'quiet': True, 'no_warnings': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info.get('id')
        except Exception:
            return None 