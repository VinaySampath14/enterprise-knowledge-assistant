from fastapi import FastAPI

app = FastAPI(title="Enterprise Knowledge Assistant")

@app.get("/health")
def health():
    return {"status": "ok"}
