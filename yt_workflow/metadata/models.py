from django.db import models

from yt_workflow.shared.models import BaseResult


class MetadataResult(BaseResult):
    """Metadata processing results"""

    raw_data = models.JSONField(default=dict, blank=True)
    processed_data = models.JSONField(default=dict, blank=True)

    class Meta:
        verbose_name = "Metadata Result"
        verbose_name_plural = "Metadata Results"
        db_table = "metadata_result"
