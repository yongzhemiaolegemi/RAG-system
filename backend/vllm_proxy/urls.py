from django.urls import path
from . import views

urlpatterns = [
    path('completions', views.vllm_proxy, name='completions'),
] 