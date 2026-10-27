"""
tests.py — Comprehensive Automated Test Suite for Weather Forecasting Django App
===============================================================================

Tests:
1. WeatherModelService: Singleton integrity, model loading, precision matching, LRU caching.
2. WeatherPredictionForm: Bounds validation, physical limits, cross-field max>=min.
3. IndexView: GET renders form, POST valid renders prediction and session history, POST invalid returns errors.
4. ApiPredictView: POST JSON returns 200 schema, invalid payload returns 400 with field errors, GET returns 405.
5. ModelSelection: Random Forest vs Ridge selection parity.
"""

import json
from django.test import Client, TestCase
from django.urls import reverse
from forecast.forms import WeatherPredictionForm
from forecast.services.ml_service import WeatherModelService


class WeatherModelServiceTest(TestCase):
    def setUp(self):
        self.service = WeatherModelService.get_instance()
        self.service.load_models()
        self.sample_features = {
            "temp_max_c": 30.5,
            "temp_min_c": 19.0,
            "humidity_pct": 52.0,
            "wind_speed_max_kmh": 16.5,
            "precipitation_mm": 0.0,
            "cloud_cover_pct": 25.0,
        }

    def test_singleton_identity(self):
        """Ensure get_instance returns the exact same object reference."""
        another_service = WeatherModelService.get_instance()
        self.assertIs(self.service, another_service)

    def test_models_are_loaded(self):
        """Verify models and metadata are resident in memory."""
        self.assertTrue(self.service.is_loaded)
        self.assertIn("rf", self.service._models)
        self.assertIn("ridge", self.service._models)
        self.assertIn("features", self.service.metadata)

    def test_prediction_output_structure_and_types(self):
        """Verify inference returns all required fields and reasonable ranges."""
        res = self.service.predict(self.sample_features, model_key="rf")
        self.assertIn("prediction", res)
        self.assertIn("prediction_exact", res)
        self.assertEqual(res["units"], "°C")
        self.assertIn("prediction_interval", res)
        self.assertIn("weather_condition", res)
        # Prediction must be between min and max temp
        self.assertGreaterEqual(res["prediction"], 19.0)
        self.assertLessEqual(res["prediction"], 30.5)

    def test_prediction_precision_match_standalone(self):
        """
        Verify web service prediction matches standalone model output to the decimal.
        From train_models.py benchmark: 24.23°C for RF, 24.53°C for Ridge.
        """
        res_rf = self.service.predict(self.sample_features, model_key="rf")
        res_ridge = self.service.predict(self.sample_features, model_key="ridge")
        self.assertAlmostEqual(res_rf["prediction_exact"], 24.2267, places=3)
        self.assertAlmostEqual(res_ridge["prediction_exact"], 24.5281, places=3)

    def test_in_memory_caching(self):
        """Repeated inference with identical inputs should trigger cache hit."""
        # First call
        res1 = self.service.predict(self.sample_features, model_key="rf")
        # Second call with identical inputs
        res2 = self.service.predict(self.sample_features, model_key="rf")
        self.assertEqual(res1["prediction"], res2["prediction"])
        self.assertTrue(res2["is_cached"])


class WeatherPredictionFormTest(TestCase):
    def test_valid_form(self):
        """Form with valid Cairo weather inputs must pass validation."""
        data = {
            "model_choice": "rf",
            "temp_max_c": 32.0,
            "temp_min_c": 21.0,
            "humidity_pct": 50.0,
            "wind_speed_max_kmh": 15.0,
            "precipitation_mm": 0.0,
            "cloud_cover_pct": 20.0,
        }
        form = WeatherPredictionForm(data=data)
        self.assertTrue(form.is_valid(), form.errors)

    def test_cross_field_validation_max_less_than_min(self):
        """temp_max_c < temp_min_c must raise a cross-field validation error."""
        data = {
            "model_choice": "rf",
            "temp_max_c": 18.0,
            "temp_min_c": 25.0,  # Min > Max is physically invalid
            "humidity_pct": 50.0,
            "wind_speed_max_kmh": 15.0,
            "precipitation_mm": 0.0,
            "cloud_cover_pct": 20.0,
        }
        form = WeatherPredictionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn("temp_max_c", form.errors)
        self.assertIn("temp_min_c", form.errors)

    def test_humidity_bounds(self):
        """Humidity > 100% or < 0% must fail validation."""
        data_high = {
            "model_choice": "rf",
            "temp_max_c": 30.0,
            "temp_min_c": 20.0,
            "humidity_pct": 105.0,
            "wind_speed_max_kmh": 10.0,
            "precipitation_mm": 0.0,
            "cloud_cover_pct": 10.0,
        }
        form = WeatherPredictionForm(data=data_high)
        self.assertFalse(form.is_valid())
        self.assertIn("humidity_pct", form.errors)

    def test_negative_precipitation(self):
        """Precipitation cannot be negative."""
        data = {
            "model_choice": "rf",
            "temp_max_c": 30.0,
            "temp_min_c": 20.0,
            "humidity_pct": 50.0,
            "wind_speed_max_kmh": 10.0,
            "precipitation_mm": -5.0,
            "cloud_cover_pct": 10.0,
        }
        form = WeatherPredictionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn("precipitation_mm", form.errors)

    def test_non_numeric_input(self):
        """Non-numeric string in numeric field must fail gracefully."""
        data = {
            "model_choice": "rf",
            "temp_max_c": "thirty",
            "temp_min_c": 20.0,
            "humidity_pct": 50.0,
            "wind_speed_max_kmh": 10.0,
            "precipitation_mm": 0.0,
            "cloud_cover_pct": 10.0,
        }
        form = WeatherPredictionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn("temp_max_c", form.errors)


class IndexViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_get_index_renders_form(self):
        """GET request renders 200 OK with form and no prediction result."""
        response = self.client.get(reverse("forecast:index"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "forecast/index.html")
        self.assertIn("form", response.context)
        self.assertIsNone(response.context["prediction"])

    def test_post_valid_form_renders_prediction(self):
        """POST valid inputs renders 200 OK with prediction panel and stores history."""
        data = {
            "model_choice": "rf",
            "temp_max_c": 32.0,
            "temp_min_c": 20.0,
            "humidity_pct": 45.0,
            "wind_speed_max_kmh": 15.0,
            "precipitation_mm": 0.0,
            "cloud_cover_pct": 10.0,
        }
        response = self.client.post(reverse("forecast:index"), data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.context["prediction"])
        self.assertGreater(response.context["prediction"]["prediction"], 20.0)
        # Check session history
        self.assertIn("prediction_history", self.client.session)
        self.assertGreaterEqual(len(self.client.session["prediction_history"]), 1)

    def test_post_invalid_form_renders_errors(self):
        """POST with invalid inputs renders 200 OK with errors and no prediction."""
        data = {
            "model_choice": "rf",
            "temp_max_c": 15.0,
            "temp_min_c": 25.0,  # Invalid: max < min
            "humidity_pct": 45.0,
            "wind_speed_max_kmh": 15.0,
            "precipitation_mm": 0.0,
            "cloud_cover_pct": 10.0,
        }
        response = self.client.post(reverse("forecast:index"), data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.context["prediction"])
        self.assertTrue(response.context["form"].errors)


class ApiPredictViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.api_url = reverse("forecast:api_predict")
        self.valid_payload = {
            "model_choice": "rf",
            "temp_max_c": 30.5,
            "temp_min_c": 19.0,
            "humidity_pct": 52.0,
            "wind_speed_max_kmh": 16.5,
            "precipitation_mm": 0.0,
            "cloud_cover_pct": 25.0,
        }

    def test_post_valid_json_returns_200(self):
        """Valid JSON payload returns 200 with complete prediction schema."""
        response = self.client.post(
            self.api_url,
            data=json.dumps(self.valid_payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["units"], "°C")
        self.assertIn("prediction", data)
        self.assertIn("prediction_interval", data)
        self.assertIn("weather_condition", data)
        self.assertAlmostEqual(data["prediction"], 24.23, places=1)

    def test_post_invalid_json_returns_400(self):
        """Invalid bounds return 400 with structured field errors."""
        invalid_payload = dict(self.valid_payload)
        invalid_payload["humidity_pct"] = 150.0  # Invalid
        response = self.client.post(
            self.api_url,
            data=json.dumps(invalid_payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertIn("humidity_pct", data["errors"])

    def test_get_request_returns_405(self):
        """GET request to predict API returns 405 Method Not Allowed."""
        response = self.client.get(self.api_url)
        self.assertEqual(response.status_code, 405)

    def test_model_selection_ridge_vs_rf(self):
        """API model_choice switch selects Ridge and returns alternative model output."""
        ridge_payload = dict(self.valid_payload)
        ridge_payload["model_choice"] = "ridge"
        response = self.client.post(
            self.api_url,
            data=json.dumps(ridge_payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["model_key"], "ridge")
        self.assertIn("Ridge", data["model_used"])
