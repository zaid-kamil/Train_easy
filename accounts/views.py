from django.shortcuts import render, redirect
from django.contrib.auth.models import User, Group
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .forms import ProfileForm , DashboardForm
from django.contrib.auth.decorators import login_required
import os
# Create your views here.

def clogin(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        group = request.POST.get('group') # customer
        if username and password:
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
            else:
                return redirect('dashboard')
        else:
            messages.error(request, 'Please fill all fields')
    return render(request, 'accounts/customer/login.html')

def cregister(request):
    if request.method == "POST":
        username = request.POST['username']
        email = request.POST['email']
        pwd1 = request.POST['password1']
        pwd2 = request.POST['password2']
        group = request.POST['group']
        if pwd1 == pwd2:
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username already exists')
            elif User.objects.filter(email=email).exists():
                messages.error(request, 'Email already exists')
            else:
                user = User.objects.create_user(username=username, email=email, password=pwd1)
                user.save()
                messages.success(request, 'Account successfully created')
                return redirect('clogin')
        else:
            messages.error(request, 'Passwords do not match')
    return render(request, 'accounts/customer/register.html')


def logout_view(request):
    logout(request)
    return redirect('home')


@login_required
def create_profile(request):
    if request.method == 'POST':
        form = ProfileForm(request.POST, request.FILES)
        if form.is_valid():
            profile = form.save(commit=False)
            profile.user = request.user
            profile.save()
            return redirect('home')
    else:
        form = ProfileForm()
    return render(request, 'accounts/create_profile.html', {'form': form})

@login_required
def dashboard(request):
    if request.method == 'POST':
        form = DashboardForm(request.POST)
        if form.is_valid():
            dashboard = form.save(commit=False)
            dashboard.user = request.user
            dashboard.save()
            return redirect('home')
    else:
        form = DashboardForm()
    return render(request, 'accounts/dashboard.html')


