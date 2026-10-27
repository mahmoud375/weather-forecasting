"""
forms.py — Django Form for Weather Prediction Input Collection & Validation
===========================================================================

Implements strict physical boundary validation and cross-field validation:
- Realistic Cairo temperature bounds (-5°C to 55°C)
- Relative humidity bounds (0% to 100%)
- Non-negative precipitation and wind speed
- Cross-field consistency: temp_max_c >= temp_min_c
- Automatic Bootstrap 5 validation state classes (is-invalid)
- Retains user-submitted values on validation failure
"""

from django import forms
from django.core.exceptions import ValidationError


class WeatherPredictionForm(forms.Form):
    MODEL_CHOICES = [
        ("rf", "Random Forest Regressor (Primary — Non-Linear)"),
        ("ridge", "Ridge Regression Pipeline (Alternative — L2 Regularized)"),
    ]

    model_choice = forms.ChoiceField(
        choices=MODEL_CHOICES,
        initial="rf",
        required=False,
        label="Forecasting Model",
        widget=forms.Select(attrs={
            "class": "form-select form-select-lg shadow-sm",
            "id": "id_model_choice",
        }),
        help_text="Select between the primary ensemble model or the linear baseline.",
    )

    def clean_model_choice(self):
        choice = self.cleaned_data.get("model_choice")
        return choice if choice in ("rf", "ridge") else "rf"


    temp_max_c = forms.FloatField(
        label="Maximum Temperature",
        min_value=-5.0,
        max_value=55.0,
        initial=30.5,
        widget=forms.NumberInput(attrs={
            "class": "form-control",
            "placeholder": "e.g. 30.5",
            "step": "0.1",
            "id": "id_temp_max_c",
        }),
        help_text="Realistic Cairo bounds: -5.0 °C to 55.0 °C",
        error_messages={
            "required": "Maximum temperature is required.",
            "invalid": "Enter a valid numeric temperature value.",
            "min_value": "Maximum temperature cannot be below -5.0 °C for Cairo.",
            "max_value": "Maximum temperature cannot exceed 55.0 °C for Cairo.",
        },
    )

    temp_min_c = forms.FloatField(
        label="Minimum Temperature",
        min_value=-5.0,
        max_value=55.0,
        initial=19.0,
        widget=forms.NumberInput(attrs={
            "class": "form-control",
            "placeholder": "e.g. 19.0",
            "step": "0.1",
            "id": "id_temp_min_c",
        }),
        help_text="Realistic Cairo bounds: -5.0 °C to 55.0 °C",
        error_messages={
            "required": "Minimum temperature is required.",
            "invalid": "Enter a valid numeric temperature value.",
            "min_value": "Minimum temperature cannot be below -5.0 °C for Cairo.",
            "max_value": "Minimum temperature cannot exceed 55.0 °C for Cairo.",
        },
    )

    humidity_pct = forms.FloatField(
        label="Relative Humidity",
        min_value=0.0,
        max_value=100.0,
        initial=52.0,
        widget=forms.NumberInput(attrs={
            "class": "form-control",
            "placeholder": "e.g. 52",
            "step": "0.5",
            "id": "id_humidity_pct",
        }),
        help_text="Physical bounds: 0.0% to 100.0%",
        error_messages={
            "required": "Relative humidity is required.",
            "invalid": "Enter a valid numeric percentage.",
            "min_value": "Relative humidity cannot be negative.",
            "max_value": "Relative humidity cannot exceed 100.0 percent.",
        },
    )

    wind_speed_max_kmh = forms.FloatField(
        label="Maximum Wind Speed",
        min_value=0.0,
        max_value=150.0,
        initial=16.5,
        widget=forms.NumberInput(attrs={
            "class": "form-control",
            "placeholder": "e.g. 16.5",
            "step": "0.1",
            "id": "id_wind_speed_max_kmh",
        }),
        help_text="Physical bounds: 0.0 to 150.0 km/h",
        error_messages={
            "required": "Wind speed is required.",
            "invalid": "Enter a valid numeric wind speed.",
            "min_value": "Wind speed cannot be negative.",
            "max_value": "Wind speed cannot exceed 150.0 km/h.",
        },
    )

    precipitation_mm = forms.FloatField(
        label="Precipitation",
        min_value=0.0,
        max_value=300.0,
        initial=0.0,
        widget=forms.NumberInput(attrs={
            "class": "form-control",
            "placeholder": "e.g. 0.0",
            "step": "0.1",
            "id": "id_precipitation_mm",
        }),
        help_text="Physical bounds: 0.0 to 300.0 mm",
        error_messages={
            "required": "Precipitation is required.",
            "invalid": "Enter a valid numeric precipitation amount.",
            "min_value": "Precipitation cannot be negative.",
            "max_value": "Precipitation cannot exceed 300.0 mm.",
        },
    )

    cloud_cover_pct = forms.FloatField(
        label="Cloud Cover",
        min_value=0.0,
        max_value=100.0,
        initial=25.0,
        widget=forms.NumberInput(attrs={
            "class": "form-control",
            "placeholder": "e.g. 25",
            "step": "1.0",
            "id": "id_cloud_cover_pct",
        }),
        help_text="Physical bounds: 0.0% to 100.0%",
        error_messages={
            "required": "Cloud cover is required.",
            "invalid": "Enter a valid numeric cloud cover percentage.",
            "min_value": "Cloud cover cannot be negative.",
            "max_value": "Cloud cover cannot exceed 100.0 percent.",
        },
    )

    def clean(self):
        """Cross-field physical consistency validation."""
        cleaned_data = super().clean()
        t_max = cleaned_data.get("temp_max_c")
        t_min = cleaned_data.get("temp_min_c")

        if t_max is not None and t_min is not None:
            if t_max < t_min:
                self.add_error(
                    "temp_max_c",
                    ValidationError(
                        f"Maximum temperature ({t_max:.1f} °C) cannot be less than minimum temperature ({t_min:.1f} °C).",
                        code="max_less_than_min",
                    ),
                )
                self.add_error(
                    "temp_min_c",
                    ValidationError(
                        f"Minimum temperature ({t_min:.1f} °C) cannot be greater than maximum temperature ({t_max:.1f} °C).",
                        code="min_greater_than_max",
                    ),
                )

        return cleaned_data
