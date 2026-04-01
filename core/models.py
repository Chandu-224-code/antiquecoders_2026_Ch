from django.db import models
from django.contrib.auth.models import User

INDIAN_LANGUAGES = [
    ('hi', 'Hindi'), ('te', 'Telugu'), ('bn', 'Bengali'), ('mr', 'Marathi'),
    ('ta', 'Tamil'), ('ur', 'Urdu'), ('gu', 'Gujarati'), ('kn', 'Kannada'),
    ('or', 'Odia'), ('ml', 'Malayalam'), ('pa', 'Punjabi'), ('as', 'Assamese'),
    ('mai', 'Maithili'), ('sat', 'Santali'), ('kas', 'Kashmiri'), ('ne', 'Nepali'),
    ('kok', 'Konkani'), ('sd', 'Sindhi'), ('doi', 'Dogri'), ('mni', 'Manipuri'),
    ('brx', 'Bodo'), ('sa', 'Sanskrit'), ('bh', 'Bhojpuri'), ('mag', 'Magahi'),
    ('awa', 'Awadhi'), ('raj', 'Rajasthani'), ('chg', 'Chhattisgarhi'), ('bgc', 'Haryanvi'),
    ('mwr', 'Marwari'),
]

FOREIGN_LANGUAGES = [
    ('en', 'English'), ('zh', 'Mandarin Chinese'), ('es', 'Spanish'), ('ar', 'Arabic'),
    ('fr', 'French'), ('ru', 'Russian'), ('pt', 'Portuguese'), ('de', 'German'),
    ('ja', 'Japanese'), ('ko', 'Korean'), ('it', 'Italian'), ('tr', 'Turkish'),
    ('vi', 'Vietnamese'), ('pl', 'Polish'), ('uk', 'Ukrainian'), ('nl', 'Dutch'),
    ('th', 'Thai'), ('sv', 'Swedish'), ('fil', 'Filipino'), ('id', 'Indonesian'),
]

LANGUAGE_CHOICES = INDIAN_LANGUAGES + FOREIGN_LANGUAGES

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    phone_number = models.CharField(max_length=20, blank=True, null=True)
    language = models.CharField(max_length=10, choices=LANGUAGE_CHOICES, default='en')

    def __str__(self):
        return f"{self.user.username}'s Profile"

class ChatQuery(models.Model):
    session_key = models.CharField(max_length=40, db_index=True)
    prompt = models.TextField()
    response = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.prompt[:50]}..."
    
    class Meta:
        ordering = ['-created_at']

