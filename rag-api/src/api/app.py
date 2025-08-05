from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import logging
import time
from routers.qa_router import router as qa_router


logging.basicConfig(level=logging.DEBUG)


app = FastAPI(docs_url="/docs")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

app.include_router(qa_router)


@app.get("/health")
def health():
    return JSONResponse(status_code=200, content={"status": "healthy"})

if __name__ == "__main__":
    uvicorn.run(app=app, host="localhost", port=8080,log_level="debug")
    # uvicorn app:app --host localhost --port 8080 --reload --log-level debug