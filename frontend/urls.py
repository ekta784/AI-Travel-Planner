from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('blog/', views.blog, name='blog'),
    path('blog-single/', views.blog_single, name='blog_single'),
    path('destination/', views.destination, name='destination'),
    path('hotel/', views.hotel, name='hotel'),
    path('main/', views.main_page, name='main'),
    path('rishikesh/', views.rishikesh, name='rishikesh'),
]
