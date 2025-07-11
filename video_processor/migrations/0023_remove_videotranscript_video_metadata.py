# Generated by Django 5.2.3 on 2025-07-10 19:10

from django.db import migrations


def noop_migration(apps, schema_editor):
    """
    No-op migration since the video_metadata field was already removed
    in the previous migration (0022_fix_foreign_key_column_types).
    """
    pass


class Migration(migrations.Migration):

    dependencies = [
        ('video_processor', '0022_fix_foreign_key_column_types'),
    ]

    operations = [
        migrations.RunPython(noop_migration, noop_migration),
    ]
