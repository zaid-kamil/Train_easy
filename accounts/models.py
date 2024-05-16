from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Profile(models.Model):
    gender_choices = [
        ('male', "Male"),
        ('female', "Female"),
        ('other', "Dont want to disclose")
    ]
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    picture= models.ImageField(upload_to='profile/')
    gender = models.CharField(max_length=10, choices=gender_choices)
    bio = models.TextField()
    user = models.OneToOneField(User, on_delete=models.CASCADE, default=1)
    
class Dashboard(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
