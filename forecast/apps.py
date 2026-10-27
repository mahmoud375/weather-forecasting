import logging
from django.apps import AppConfig

logger = logging.getLogger("forecast.apps")


class ForecastConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "forecast"

    def ready(self):
        """
        Executed once when Django starts up.
        Initializes and warms up the WeatherModelService singleton so model
        weights reside in memory before the first HTTP request arrives.
        """
        # Avoid running model loading during management commands like makemigrations or collectstatic
        import sys
        if any(cmd in sys.argv for cmd in ["makemigrations", "migrate", "collectstatic"]):
            return

        try:
            from forecast.services.ml_service import WeatherModelService
            service = WeatherModelService.get_instance()
            service.load_models()
            logger.info("ForecastConfig.ready: Weather models loaded and ready for inference.")
        except Exception as exc:
            logger.warning(f"ForecastConfig.ready: Deferred model loading ({exc})")
