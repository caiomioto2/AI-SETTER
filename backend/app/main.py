from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api import webhooks

app = FastAPI(title="AI Setter API", description="AI-powered lead engagement and booking automation via GoHighLevel webhooks.", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(webhooks.router, prefix="/webhooks", tags=["Webhooks"])

@app.get("/health")
async def health_check():
    return {"status": "ok"}
