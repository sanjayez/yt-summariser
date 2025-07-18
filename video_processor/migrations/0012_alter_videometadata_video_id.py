# Generated by Django 5.2.3 on 2025-07-10 02:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('video_processor', '0011_populate_missing_video_ids'),
    ]

    operations = [
        migrations.AlterField(
            model_name='videometadata',
            name='video_id',
            field=models.CharField(db_index=True, default='default_video_id', help_text="YouTube video ID (e.g., 'dQw4w9WgXcQ')", max_length=20, unique=True),
            preserve_default=False,
        ),
    ]
