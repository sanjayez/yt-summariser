# Generated by Django 5.2.3 on 2025-07-18 08:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('topic', '0001_initial'),
    ]

    operations = [
        # Rename the field from request_id to search_id
        migrations.RenameField(
            model_name='searchrequest',
            old_name='request_id',
            new_name='search_id',
        ),
        # Remove the old index on request_id
        migrations.RemoveIndex(
            model_name='searchrequest',
            name='topic_searc_request_061e1d_idx',
        ),
        # Add new index on search_id
        migrations.AddIndex(
            model_name='searchrequest',
            index=models.Index(fields=['search_id'], name='topic_searc_search__search_id_idx'),
        ),
    ]
