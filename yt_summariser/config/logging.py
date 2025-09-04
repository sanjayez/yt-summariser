"""
Centralized logging configuration for YT Summariser.
"""


def get_logging_config(debug: bool = False) -> dict:
    """
    Get Django LOGGING configuration.

    Args:
        debug: Whether running in debug/development mode

    Returns:
        Complete Django LOGGING configuration dictionary
    """
    # Determine formatters based on environment
    file_formatter = "detailed" if debug else "production"

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "development": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "production": {
                "()": "telemetry.logging.formatters.JSONFormatter",
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "development",
                "level": "INFO",
            },
            "django_file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "filename": "logs/application/django.log",
                "when": "midnight",
                "interval": 1,
                "backupCount": 30,
                "formatter": file_formatter,
                "encoding": "utf-8",
            },
            "celery_file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "filename": "logs/celery/celery.log",
                "when": "midnight",
                "interval": 1,
                "backupCount": 30,
                "formatter": file_formatter,
                "encoding": "utf-8",
            },
            "performance_file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "filename": "logs/monitoring/performance.log",
                "when": "midnight",
                "interval": 1,
                "backupCount": 7,
                "formatter": "detailed",
                "encoding": "utf-8",
            },
            "unified_json": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "filename": "logs/unified/all-services.jsonl",
                "when": "midnight",
                "interval": 1,
                "backupCount": 30,
                "formatter": "production",
                "encoding": "utf-8",
            },
        },
        "loggers": {
            # Django framework logs
            "django": {
                "handlers": ["console", "django_file", "unified_json"],
                "level": "INFO",
                "propagate": False,
            },
            "django.request": {
                "handlers": ["console", "django_file"],
                "level": "WARNING",
                "propagate": False,
            },
            "django.db.backends": {
                "handlers": ["django_file"] if debug else [],
                "level": "DEBUG" if debug else "WARNING",
                "propagate": False,
            },
            # Application modules
            "query_processor": {
                "handlers": ["console", "celery_file", "unified_json"],
                "level": "INFO",
                "propagate": True,
            },
            "ai_utils": {
                "handlers": ["console", "celery_file", "unified_json"],
                "level": "INFO",
                "propagate": False,
            },
            "api": {
                "handlers": ["console", "django_file", "unified_json"],
                "level": "INFO",
                "propagate": False,
            },
            # Telemetry and monitoring
            "telemetry": {
                "handlers": ["console", "performance_file"],
                "level": "INFO",
                "propagate": False,
            },
            # Celery framework logs
            "celery": {
                "handlers": ["console", "celery_file", "unified_json"],
                "level": "INFO",
                "propagate": False,
            },
            "celery.task": {
                "handlers": ["console", "celery_file"],
                "level": "INFO",
                "propagate": False,
            },
            "celery.worker": {
                "handlers": ["celery_file"],
                "level": "INFO",
                "propagate": False,
            },
            # Third-party libraries (quieter)
            "urllib3": {
                "handlers": ["unified_json"],
                "level": "WARNING",
                "propagate": False,
            },
            "requests": {
                "handlers": ["unified_json"],
                "level": "WARNING",
                "propagate": False,
            },
            "openai": {
                "handlers": ["celery_file", "unified_json"],
                "level": "WARNING",
                "propagate": False,
            },
            "google": {
                "handlers": ["celery_file", "unified_json"],
                "level": "WARNING",
                "propagate": False,
            },
        },
        "root": {
            "handlers": ["console", "unified_json"],
            "level": "WARNING",
        },
    }


def get_celery_logging_config() -> dict:
    """
    Get Celery-specific logging configuration.

    Returns:
        Celery logging configuration for worker processes
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "celery": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - [Worker:%(processName)s] - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "celery_json": {
                "()": "telemetry.logging.formatters.JSONFormatter",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "celery",
            },
            "file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "filename": "logs/celery/worker.log",
                "when": "midnight",
                "interval": 1,
                "backupCount": 30,
                "formatter": "celery_json",
                "encoding": "utf-8",
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": "INFO",
        },
    }
