"""
views.py — Slim Django Controllers for Web UI and REST API
==========================================================

Separation of concerns:
- index_view: Renders form on GET, validates and invokes ML service on POST.
  Maintains session-backed history for Chart.js visualization.
- api_predict_view: JSON REST API endpoint at /api/predict/ accepting POST
  payloads and returning structured inference results.
"""

from datetime import datetime
import json
import logging
from typing import Any, Dict, List
from django.http import HttpRequest, HttpResponse, HttpResponseNotAllowed, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from forecast.forms import WeatherPredictionForm
from forecast.services.ml_service import WeatherModelService

logger = logging.getLogger("forecast.views")


def index_view(request: HttpRequest) -> HttpResponse:
    """
    Main web application dashboard.
    Handles GET (render form with Cairo default placeholders) and
    POST (validate input, predict target, and display result panel).
    """
    service = WeatherModelService.get_instance()
    metadata = service.metadata

    # Retrieve or initialize session prediction history
    history: List[Dict[str, Any]] = request.session.get("prediction_history", [])

    prediction_result = None

    if request.method == "POST":
        form = WeatherPredictionForm(request.POST)
        if form.is_valid():
            model_key = form.cleaned_data["model_choice"]
            features = {
                "temp_max_c": form.cleaned_data["temp_max_c"],
                "temp_min_c": form.cleaned_data["temp_min_c"],
                "humidity_pct": form.cleaned_data["humidity_pct"],
                "wind_speed_max_kmh": form.cleaned_data["wind_speed_max_kmh"],
                "precipitation_mm": form.cleaned_data["precipitation_mm"],
                "cloud_cover_pct": form.cleaned_data["cloud_cover_pct"],
            }

            prediction_result = service.predict(features, model_key=model_key)
            prediction_result["inputs"] = features

            # Append to session history (capped at last 10 entries)
            history_entry = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "prediction": prediction_result["prediction"],
                "model": "Random Forest" if model_key == "rf" else "Ridge Regression",
                "condition": prediction_result["weather_condition"]["condition"],
                "temp_max": features["temp_max_c"],
                "temp_min": features["temp_min_c"],
                "humidity": features["humidity_pct"],
            }
            history.insert(0, history_entry)
            request.session["prediction_history"] = history[:10]
            request.session.modified = True
            logger.info(f"Generated web prediction: {prediction_result['prediction']}°C ({model_key})")
    else:
        # Prepopulate with representative Cairo day if no prior POST
        form = WeatherPredictionForm(initial=metadata.get("default_cairo_example", {}))

    context = {
        "form": form,
        "prediction": prediction_result,
        "metadata": metadata,
        "history": history[:8],
        "history_json": json.dumps(history[:8]),
    }
    return render(request, "forecast/index.html", context)


@csrf_exempt
def api_predict_view(request: HttpRequest) -> JsonResponse:
    """
    Programmatic REST API Endpoint: POST /api/predict/
    Accepts JSON body:
    {
        "model_choice": "rf" | "ridge",  # optional, default "rf"
        "temp_max_c": 30.5,
        "temp_min_c": 19.0,
        "humidity_pct": 52.0,
        "wind_speed_max_kmh": 16.5,
        "precipitation_mm": 0.0,
        "cloud_cover_pct": 25.0
    }
    Returns 200 JSON on success, 400 on validation error, 405 on non-POST.
    """
    if request.method != "POST":
        return JsonResponse(
            {
                "status": "error",
                "message": f"Method {request.method} not allowed. Send a POST request.",
            },
            status=405,
        )

    try:
        data = json.loads(request.body.decode("utf-8")) if request.body else {}
    except json.JSONDecodeError as exc:
        return JsonResponse(
            {
                "status": "error",
                "message": f"Malformed JSON payload: {str(exc)}",
            },
            status=400,
        )

    form = WeatherPredictionForm(data)
    if not form.is_valid():
        formatted_errors = {
            field: [err["message"] for err in err_list]
            for field, err_list in form.errors.get_json_data().items()
        }
        return JsonResponse(
            {
                "status": "error",
                "message": "Input validation failed.",
                "errors": formatted_errors,
            },
            status=400,
        )

    model_key = form.cleaned_data.get("model_choice", "rf")
    features = {
        "temp_max_c": form.cleaned_data["temp_max_c"],
        "temp_min_c": form.cleaned_data["temp_min_c"],
        "humidity_pct": form.cleaned_data["humidity_pct"],
        "wind_speed_max_kmh": form.cleaned_data["wind_speed_max_kmh"],
        "precipitation_mm": form.cleaned_data["precipitation_mm"],
        "cloud_cover_pct": form.cleaned_data["cloud_cover_pct"],
    }

    service = WeatherModelService.get_instance()
    result = service.predict(features, model_key=model_key)

    response_payload = {
        "status": "success",
        "prediction": result["prediction"],
        "units": result["units"],
        "model_used": result["model_used"],
        "model_key": result["model_key"],
        "prediction_interval": result["prediction_interval"],
        "weather_condition": result["weather_condition"],
        "is_cached": result["is_cached"],
        "inputs": features,
    }
    return JsonResponse(response_payload, status=200)
