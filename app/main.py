from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import get_settings
from app.core.exceptions import MedPilotError


settings = get_settings()
app = FastAPI(title=settings.app_name)
app.include_router(router)


@app.exception_handler(MedPilotError)
async def handle_domain_error(_, exc: MedPilotError):
    return JSONResponse(status_code=400, content={"ok": False, "error": str(exc)})


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"ok": False, "error": exc.errors()})
