from django import forms

from .models import Profile , Dashboard

class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['picture', 'first_name', 'last_name', 'gender',  'bio']

class DashboardForm(forms.ModelForm):
    class Meta:
        model = Dashboard
        fields = ['is_active']
        
