from fastapi import FastAPI
from domain_services.invoice_api import router as invoice_router
from domain_services.explain_api import router as explain_router

app = FastAPI(
    title="Intent-Driven Enterprise Assistant",
    version="0.1.0"
)

app.include_router(invoice_router, prefix="/api")
app.include_router(explain_router, prefix="/api")
