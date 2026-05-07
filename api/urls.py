from django.urls import path
from api import views

urlpatterns = [
    path('rate/', views.rate, name='rate'),
    path('reset/', views.reset_ratings, name='reset_ratings'),
]