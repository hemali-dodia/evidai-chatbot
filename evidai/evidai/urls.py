from django.contrib import admin
from django.urls import path
from evidai_chat import views

urlpatterns = [
    path('',views.hello_world,name='hello_world'),
    path('chat',views.get_gemini_response,name="get_gemini_response"),
    path('create_chat_session/',views.create_chat_session,name='create_chat_session'),
    path('get_conversations',views.get_conversations,name='get_conversations'),
    path('evidAI_chat',views.evidAI_chat,name='evidAI_chat'),
    path('generate_token', views.create_token, name='generate_token'),
    path('delete_chat_session',views.delete_chat_session,name='delete_chat_session'),
    path('get_chat_session_details',views.get_chat_session_details,name='get_chat_session_details')
]
