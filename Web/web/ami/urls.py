from django.urls import path
from .views import c_test,test


urlpatterns = [
    path("", test),
    path('pred', c_test, name="predict")

]