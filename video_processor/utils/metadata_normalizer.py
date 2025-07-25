"""
YouTube Metadata Normalization Layer

This module provides centralized normalization, validation, and default value application
for yt-dlp metadata to ensure consistent data structure throughout the video processing pipeline.

Key functions:
- Normalize raw yt-dlp metadata to consistent format
- Apply appropriate default values for missing fields
- Validate field constraints and data types
- Handle edge cases and data inconsistencies
"""

import logging
from datetime import datetime
from typing import Dict, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class YouTubeMetadataNormalizer:
    """Centralized yt-dlp metadata normalization and validation"""
    
    # Field constraints from model definitions
    FIELD_CONSTRAINTS = {
        'video_id': {'max_length': 20, 'required': True},
        'title': {'max_length': 500, 'default': ''},
        'description': {'default': ''},
        'duration': {'min_value': 0, 'max_value': 86400, 'default': None},  # 24 hours max
        'channel_name': {'max_length': 200, 'default': ''},
        'view_count': {'min_value': 0, 'default': 0},
        'like_count': {'min_value': 0, 'default': 0},
        'channel_follower_count': {'min_value': 0, 'default': 0},
        'language': {'max_length': 12, 'default': 'fallback-en'},
        'channel_id': {'max_length': 50, 'default': ''},
        'uploader_id': {'max_length': 100, 'default': ''},
        'tags': {'max_items': 50, 'default': []},
        'categories': {'max_items': 10, 'default': []},
        'thumbnail': {'default': ''},
        'channel_is_verified': {'default': False},
    }
    
    # Common language code mappings for normalization
    LANGUAGE_MAPPINGS = {
        'en-US': 'en',
        'en-GB': 'en', 
        'en-CA': 'en',
        'es-ES': 'es',
        'es-MX': 'es',
        'fr-FR': 'fr',
        'fr-CA': 'fr',
        'de-DE': 'de',
        'pt-BR': 'pt',
        'pt-PT': 'pt',
        'zh-CN': 'zh',
        'zh-TW': 'zh',
    }
    
    @classmethod
    def normalize_metadata(cls, raw_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert raw yt-dlp info to normalized, validated metadata
        
        Args:
            raw_info: Raw dictionary from yt-dlp extraction
            
        Returns:
            Normalized metadata dictionary ready for database storage
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        if not raw_info:
            raise ValueError("Raw metadata info cannot be empty")
            
        # Start with basic field extraction
        normalized = cls._extract_basic_fields(raw_info)
        
        # Apply field-specific normalizations
        normalized = cls._normalize_strings(normalized)
        normalized = cls._normalize_numeric_fields(normalized)
        normalized = cls._normalize_language(normalized, raw_info)
        normalized = cls._normalize_arrays(normalized)
        normalized = cls._normalize_dates(normalized, raw_info)
        normalized = cls._normalize_urls(normalized)
        normalized = cls._normalize_booleans(normalized)
        
        # Apply defaults for missing fields
        normalized = cls._apply_defaults(normalized)
        
        # Final validation
        cls._validate_constraints(normalized)
        
        logger.debug(f"Normalized metadata for video {normalized.get('video_id', 'unknown')}")
        return normalized
    
    @classmethod
    def _extract_basic_fields(cls, raw_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic fields from raw yt-dlp info"""
        return {
            'video_id': raw_info.get('id'),
            'title': raw_info.get('title'),
            'description': raw_info.get('description'),
            'duration': raw_info.get('duration'),
            'channel_name': raw_info.get('uploader'),
            'view_count': raw_info.get('view_count'),
            'like_count': raw_info.get('like_count'),
            'channel_follower_count': raw_info.get('uploader_subscriber_count'),
            'language': raw_info.get('language'),
            'channel_id': raw_info.get('uploader_id'),
            'uploader_id': raw_info.get('uploader_id'),
            'tags': raw_info.get('tags'),
            'categories': raw_info.get('categories'),
            'thumbnail': raw_info.get('thumbnail'),
            'channel_is_verified': raw_info.get('uploader_verified'),
            'upload_date': raw_info.get('upload_date'),
        }
    
    @classmethod
    def _normalize_strings(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize string fields with length constraints and cleaning"""
        
        # Video ID validation (required field)
        video_id = metadata.get('video_id', '').strip()
        if not video_id:
            raise ValueError("Video ID is required")
        if len(video_id) != 11:
            raise ValueError(f"Invalid YouTube video ID format: {video_id}")
        metadata['video_id'] = video_id
        
        # Title normalization with length constraint
        title = (metadata.get('title') or '').strip()
        if len(title) > cls.FIELD_CONSTRAINTS['title']['max_length']:
            logger.warning(f"Title truncated from {len(title)} to {cls.FIELD_CONSTRAINTS['title']['max_length']} chars")
            title = title[:cls.FIELD_CONSTRAINTS['title']['max_length']].strip()
        metadata['title'] = title
        
        # Channel name normalization
        channel_name = (metadata.get('channel_name') or '').strip()
        if len(channel_name) > cls.FIELD_CONSTRAINTS['channel_name']['max_length']:
            logger.warning(f"Channel name truncated from {len(channel_name)} to {cls.FIELD_CONSTRAINTS['channel_name']['max_length']} chars")
            channel_name = channel_name[:cls.FIELD_CONSTRAINTS['channel_name']['max_length']].strip()
        metadata['channel_name'] = channel_name
        
        # Channel ID normalization
        channel_id = (metadata.get('channel_id') or '').strip()
        if len(channel_id) > cls.FIELD_CONSTRAINTS['channel_id']['max_length']:
            channel_id = channel_id[:cls.FIELD_CONSTRAINTS['channel_id']['max_length']]
        metadata['channel_id'] = channel_id
        
        # Uploader ID normalization
        uploader_id = (metadata.get('uploader_id') or '').strip()
        if len(uploader_id) > cls.FIELD_CONSTRAINTS['uploader_id']['max_length']:
            uploader_id = uploader_id[:cls.FIELD_CONSTRAINTS['uploader_id']['max_length']]
        metadata['uploader_id'] = uploader_id
        
        # Description normalization (no length limit but ensure it's a string)
        description = metadata.get('description')
        if description is None:
            metadata['description'] = ''
        else:
            metadata['description'] = str(description)
        
        return metadata
    
    @classmethod
    def _normalize_numeric_fields(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize numeric fields with range validation and defaults"""
        
        # Duration normalization
        duration = metadata.get('duration')
        if duration is not None:
            try:
                duration = int(duration)
                if duration < cls.FIELD_CONSTRAINTS['duration']['min_value']:
                    logger.warning(f"Invalid duration {duration}, setting to None")
                    duration = None
                elif duration > cls.FIELD_CONSTRAINTS['duration']['max_value']:
                    logger.warning(f"Duration {duration} exceeds maximum, capping at {cls.FIELD_CONSTRAINTS['duration']['max_value']}")
                    duration = cls.FIELD_CONSTRAINTS['duration']['max_value']
            except (ValueError, TypeError):
                logger.warning(f"Invalid duration value {duration}, setting to None")
                duration = None
        metadata['duration'] = duration
        
        # View count normalization
        view_count = metadata.get('view_count')
        if view_count is not None:
            try:
                view_count = int(view_count)
                if view_count < cls.FIELD_CONSTRAINTS['view_count']['min_value']:
                    view_count = cls.FIELD_CONSTRAINTS['view_count']['default']
            except (ValueError, TypeError):
                view_count = cls.FIELD_CONSTRAINTS['view_count']['default']
        else:
            view_count = cls.FIELD_CONSTRAINTS['view_count']['default']
        metadata['view_count'] = view_count
        
        # Like count normalization
        like_count = metadata.get('like_count')
        if like_count is not None:
            try:
                like_count = int(like_count)
                if like_count < cls.FIELD_CONSTRAINTS['like_count']['min_value']:
                    like_count = cls.FIELD_CONSTRAINTS['like_count']['default']
            except (ValueError, TypeError):
                like_count = cls.FIELD_CONSTRAINTS['like_count']['default']
        else:
            like_count = cls.FIELD_CONSTRAINTS['like_count']['default']
        metadata['like_count'] = like_count
        
        # Channel follower count normalization
        follower_count = metadata.get('channel_follower_count')
        if follower_count is not None:
            try:
                follower_count = int(follower_count)
                if follower_count < cls.FIELD_CONSTRAINTS['channel_follower_count']['min_value']:
                    follower_count = cls.FIELD_CONSTRAINTS['channel_follower_count']['default']
            except (ValueError, TypeError):
                follower_count = cls.FIELD_CONSTRAINTS['channel_follower_count']['default']
        else:
            follower_count = cls.FIELD_CONSTRAINTS['channel_follower_count']['default']
        metadata['channel_follower_count'] = follower_count
        
        return metadata
    
    @classmethod
    def _normalize_language(cls, metadata: Dict[str, Any], raw_info: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize language field with unified fallback logic"""
        
        language = metadata.get('language')
        
        # Handle None or empty language
        if not language:
            # Check for automatic captions as language indicator
            auto_captions = raw_info.get('automatic_captions', {})
            if auto_captions:
                # Use first available auto caption language
                language = list(auto_captions.keys())[0]
                logger.debug(f"Using auto caption language: {language}")
            else:
                # Apply fallback logic
                language = 'fallback-en'
                logger.debug("No language detected, using fallback-en")
        
        # Normalize common language code variations
        if language in cls.LANGUAGE_MAPPINGS:
            original_language = language
            language = cls.LANGUAGE_MAPPINGS[language]
            logger.debug(f"Mapped language {original_language} to {language}")
        
        # Ensure language doesn't exceed length constraint
        if len(language) > cls.FIELD_CONSTRAINTS['language']['max_length']:
            logger.warning(f"Language code {language} too long, truncating")
            language = language[:cls.FIELD_CONSTRAINTS['language']['max_length']]
        
        metadata['language'] = language
        return metadata
    
    @classmethod
    def _normalize_arrays(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize array fields (tags, categories)"""
        
        # Tags normalization
        tags = metadata.get('tags')
        if tags is None:
            tags = []
        elif not isinstance(tags, list):
            logger.warning(f"Tags is not a list: {type(tags)}, converting to empty list")
            tags = []
        else:
            # Filter out None/empty tags and limit size
            tags = [str(tag).strip() for tag in tags if tag is not None and str(tag).strip()]
            if len(tags) > cls.FIELD_CONSTRAINTS['tags']['max_items']:
                logger.warning(f"Too many tags ({len(tags)}), limiting to {cls.FIELD_CONSTRAINTS['tags']['max_items']}")
                tags = tags[:cls.FIELD_CONSTRAINTS['tags']['max_items']]
        metadata['tags'] = tags
        
        # Categories normalization
        categories = metadata.get('categories')
        if categories is None:
            categories = []
        elif not isinstance(categories, list):
            logger.warning(f"Categories is not a list: {type(categories)}, converting to empty list")
            categories = []
        else:
            # Filter out None/empty categories and limit size
            categories = [str(cat).strip() for cat in categories if cat is not None and str(cat).strip()]
            if len(categories) > cls.FIELD_CONSTRAINTS['categories']['max_items']:
                logger.warning(f"Too many categories ({len(categories)}), limiting to {cls.FIELD_CONSTRAINTS['categories']['max_items']}")
                categories = categories[:cls.FIELD_CONSTRAINTS['categories']['max_items']]
        metadata['categories'] = categories
        
        return metadata
    
    @classmethod
    def _normalize_dates(cls, metadata: Dict[str, Any], raw_info: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize date fields with robust parsing"""
        
        upload_date_str = raw_info.get('upload_date')
        upload_date = None
        
        if upload_date_str:
            try:
                # yt-dlp typically provides dates in YYYYMMDD format
                if isinstance(upload_date_str, str) and len(upload_date_str) == 8:
                    upload_date = datetime.strptime(upload_date_str, '%Y%m%d').date()
                elif isinstance(upload_date_str, str):
                    # Try other common formats
                    for date_format in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']:
                        try:
                            upload_date = datetime.strptime(upload_date_str, date_format).date()
                            break
                        except ValueError:
                            continue
                elif isinstance(upload_date_str, (int, float)):
                    # Unix timestamp
                    upload_date = datetime.fromtimestamp(upload_date_str).date()
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse upload date '{upload_date_str}': {e}")
                upload_date = None
        
        metadata['upload_date'] = upload_date
        return metadata
    
    @classmethod
    def _normalize_urls(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize URL fields with validation"""
        
        thumbnail = (metadata.get('thumbnail') or '').strip()
        
        # Validate thumbnail URL
        if thumbnail:
            try:
                parsed = urlparse(thumbnail)
                if not parsed.scheme or not parsed.netloc:
                    logger.warning(f"Invalid thumbnail URL: {thumbnail}")
                    thumbnail = ''
            except Exception as e:
                logger.warning(f"Failed to parse thumbnail URL {thumbnail}: {e}")
                thumbnail = ''
        
        metadata['thumbnail'] = thumbnail
        return metadata
    
    @classmethod
    def _normalize_booleans(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize boolean fields"""
        
        # Channel verification status
        is_verified = metadata.get('channel_is_verified')
        if is_verified is None:
            is_verified = cls.FIELD_CONSTRAINTS['channel_is_verified']['default']
        else:
            is_verified = bool(is_verified)
        metadata['channel_is_verified'] = is_verified
        
        return metadata
    
    @classmethod
    def _apply_defaults(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values for any missing fields"""
        
        for field, constraints in cls.FIELD_CONSTRAINTS.items():
            if field not in metadata or metadata[field] is None:
                if 'default' in constraints:
                    metadata[field] = constraints['default']
                    logger.debug(f"Applied default value for field {field}: {constraints['default']}")
        
        return metadata
    
    @classmethod
    def _validate_constraints(cls, metadata: Dict[str, Any]) -> None:
        """Final validation of field constraints"""
        
        # Validate required fields
        for field, constraints in cls.FIELD_CONSTRAINTS.items():
            if constraints.get('required') and not metadata.get(field):
                raise ValueError(f"Required field {field} is missing or empty")
        
        # Validate string length constraints
        for field in ['video_id', 'title', 'channel_name', 'language', 'channel_id', 'uploader_id']:
            if field in metadata and metadata[field]:
                max_length = cls.FIELD_CONSTRAINTS[field].get('max_length')
                if max_length and len(str(metadata[field])) > max_length:
                    raise ValueError(f"Field {field} exceeds maximum length {max_length}")
        
        # Validate numeric ranges
        for field in ['duration', 'view_count', 'like_count', 'channel_follower_count']:
            if field in metadata and metadata[field] is not None:
                value = metadata[field]
                constraints = cls.FIELD_CONSTRAINTS[field]
                
                if 'min_value' in constraints and value < constraints['min_value']:
                    raise ValueError(f"Field {field} below minimum value {constraints['min_value']}")
                
                if 'max_value' in constraints and value > constraints['max_value']:
                    raise ValueError(f"Field {field} above maximum value {constraints['max_value']}")
        
        logger.debug(f"Metadata validation passed for video {metadata.get('video_id')}")


def normalize_youtube_metadata(raw_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function for metadata normalization
    
    Args:
        raw_info: Raw dictionary from yt-dlp extraction
        
    Returns:
        Normalized metadata dictionary ready for database storage
    """
    return YouTubeMetadataNormalizer.normalize_metadata(raw_info)