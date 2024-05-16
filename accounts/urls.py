from django.urls import path
from . import views

urlpatterns = [
    path("login", views.clogin, name="clogin"),  
    path("register", views.cregister, name="cregister"),
    path('logout', views.logout_view, name='logout'),
    path('profile/create', views.create_profile, name='create_profile'),
    path('dashboard/create', views.dashboard, name='dashboard'),
    
]