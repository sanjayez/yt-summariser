# Generated by Django 5.2.3 on 2025-07-03 08:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('video_processor', '0002_videometadata_status_videotranscript_status'),
    ]

    operations = [
        migrations.AddField(
            model_name='videotranscript',
            name='transcript_data',
            field=models.JSONField(blank=True, null=True),
        ),
    ]
