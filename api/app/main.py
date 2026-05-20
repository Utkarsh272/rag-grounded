# api/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.documents import router as documents_router
from app.routes.search import router as search_router
from app.routes.conversations import router as conversations_router
from app.routes.messages import router as messages_router

app = FastAPI(title="RAG with Grounded Citations")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(search_router)
app.include_router(conversations_router)
app.include_router(messages_router)

@app.get("/healthz")
def health():
    return {"status": "ok"}
