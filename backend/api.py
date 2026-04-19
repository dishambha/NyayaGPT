try:
	from backend.backend_app import app
except ModuleNotFoundError:
	from backend_app import app

__all__ = ["app"]
