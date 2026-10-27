"""URLs for forecast application."""
from django.urls import path
from forecast import views

app_name = "forecast"

urlpatterns = [
    path("", views.index_view, name="index"),
    path("api/predict/", views.api_predict_view, name="api_predict"),
]
