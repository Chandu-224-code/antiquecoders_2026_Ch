from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('new-chat/', views.new_chat, name='new_chat'),
    path('chat/<str:chat_id>/', views.load_chat, name='load_chat'),
    path('delete-chat/<str:chat_id>/', views.delete_chat, name='delete_chat'),
    # Custom account routes to support email/phone login and signup
    path('auth/google/callback', views.google_callback_view, name='google_callback_legacy'),
    path('auth/google/callback/', views.google_callback_view),
    path('accounts/login/', views.login_view, name='login'),
    path('accounts/login/google/', views.google_login_view, name='google_login'),
    path('accounts/google/callback/', views.google_callback_view, name='google_callback'),
    path('accounts/signup/', views.signup_view, name='signup'),
    path('accounts/profile/', views.account_profile_view, name='account_profile'),
    path('accounts/settings/', views.account_settings_view, name='account_settings'),
    path('accounts/logout/', views.logout_view, name='logout'),
]
