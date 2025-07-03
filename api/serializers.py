from rest_framework import serializers
from .models import URLRequestTable

class URLRequestTableSerializer(serializers.ModelSerializer):
    class Meta:
        model = URLRequestTable
        fields = '__all__'