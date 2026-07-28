import os
import time
import uuid
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from main import process_image
from src.cost_logger import CostLogger

app = FastAPI(title="Optera Document AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)
cost_logger = CostLogger()

TASKS = {}

def process_image_task(task_id: str, file_path: str, mode: str):
    def update_state(state):
        TASKS[task_id] = state
        
    TASKS[task_id] = {"stages": [], "done": False, "final_json": None}
    process_image(file_path, "output", cost_logger, update_state=update_state, mode=mode)

@app.post("/upload")
async def upload_image(background_tasks: BackgroundTasks, file: UploadFile = File(...), mode: str = "optimized"):
    # Save the file temporarily
    task_id = str(uuid.uuid4())
    file_path = os.path.join("input", f"{task_id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
        
    background_tasks.add_task(process_image_task, task_id, file_path, mode)
    return JSONResponse(content={"task_id": task_id})

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    return JSONResponse(content=TASKS[task_id])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
