# Generated by Django 5.2.3 on 2025-07-19 10:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0007_add_failed_permanently_status'),
    ]

    operations = [
        migrations.AlterField(
            model_name='urlrequesttable',
            name='status',
            field=models.CharField(choices=[('processing', 'Processing'), ('success', 'Success'), ('failed', 'Failed')], default='processing', max_length=20),
        ),
    ]
