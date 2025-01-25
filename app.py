import time
from fastapi import FastAPI, BackgroundTasks, File, HTTPException, UploadFile, Form
from typing import Dict, List, Optional
import uuid
from pathlib import Path
import asyncio
import shutil
import os
from fastapi.responses import JSONResponse, FileResponse
from library.gvhmr import GvhmrInfer
import logging
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# 任务管理相关
TASK_STATUS = ("queued", "processing", "completed", "failed")
TEMP_DIR = Path("temp")
OUTPUT_DIR = Path("outputs")

# 初始化目录
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

class TaskManager:
    def __init__(self):
        self.task_queue = asyncio.Queue()
        self.tasks: Dict[str, Dict] = {}
        self.current_workers = 0
        self.max_workers = 1  # 由于GPU资源限制，同时只处理一个任务
        
    async def process_queue(self):
        while True:
            if self.current_workers < self.max_workers:
                task_id = await self.task_queue.get()
                self.current_workers += 1
                try:
                    await self.process_task(task_id)
                finally:
                    self.current_workers -= 1
            await asyncio.sleep(1)
    
    async def process_task(self, task_id: str):
        task = self.tasks[task_id]
        try:
            task["status"] = "processing"
            logger.info(f"Processing task {task_id}")
            
            # 执行推理
            bvh_path = gvhmr_infer.infer(
                video_path=task["video_path"],
                static_cam=False,
                fps=30,
                render=False,
                output_root=str(OUTPUT_DIR / task_id)
            )
            
            # 移动结果到持久化存储
            output_path = OUTPUT_DIR / task_id / "results.bvh"
            shutil.move(bvh_path, output_path)
            
            task.update({
                "status": "completed",
                "bvh_path": str(output_path),
                "error": None
            })
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {str(e)}")
            task.update({
                "status": "failed",
                "error": str(e)
            })
        finally:
            # 清理临时文件
            shutil.rmtree(task["temp_dir"], ignore_errors=True)

task_manager = TaskManager()
gvhmr_infer = None


@app.on_event("startup")
async def startup_event():
    global gvhmr_infer
    gvhmr_infer = GvhmrInfer()
    gvhmr_infer.load_model()
    asyncio.create_task(task_manager.process_queue())

@app.post("/tasks")
async def create_task(
    video: UploadFile = File(...),
    static_cam: bool = Form(False),
    fps: int = Form(30)
) -> Dict:
    """创建新处理任务"""
    try:
        # 生成任务ID
        task_id = str(uuid.uuid4())
        temp_dir = TEMP_DIR / task_id
        temp_dir.mkdir()
        
        # 保存上传视频
        video_path = temp_dir / video.filename
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # 创建任务记录
        task_manager.tasks[task_id] = {
            "id": task_id,
            "status": "queued",
            "created_at": int(time.time()),
            "video_path": str(video_path),
            "temp_dir": str(temp_dir),
            "bvh_path": None,
            "error": None
        }
        
        # 加入处理队列
        await task_manager.task_queue.put(task_id)
        
        return JSONResponse({
            "task_id": task_id,
            "status_url": f"/tasks/{task_id}",
            "message": "Task created successfully"
        }, status_code=201)
        
    except Exception as e:
        logger.error(f"Task creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks")
async def list_tasks(
    status: Optional[str] = None, 
    limit: int = 100
) -> List[Dict]:
    """获取任务列表"""
    try:
        tasks = list(task_manager.tasks.values())
        
        # 过滤状态
        if status and status in TASK_STATUS:
            tasks = [t for t in tasks if t["status"] == status]
            
        # 按时间倒序
        tasks = sorted(tasks, key=lambda x: x["created_at"], reverse=True)[:limit]
        
        # 简化返回字段
        return [{
            "id": t["id"],
            "status": t["status"],
            "created_at": t["created_at"],
            "result_url": f"/tasks/{t['id']}/download" if t["bvh_path"] else None
        } for t in tasks]
        
    except Exception as e:
        logger.error(f"Get task list failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks/{task_id}")
async def get_task(task_id: str) -> Dict:
    """获取单个任务状态"""
    if task_id not in task_manager.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = task_manager.tasks[task_id]
    return {
        "id": task_id,
        "status": task["status"],
        "created_at": task["created_at"],
        "progress": task.get("progress", 0),
        "result_url": f"/tasks/{task_id}/download" if task["bvh_path"] else None,
        "error": task["error"]
    }

@app.get("/tasks/{task_id}/download")
async def download_result(task_id: str):
    """下载BVH结果文件"""
    if task_id not in task_manager.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = task_manager.tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=425, detail="Result not ready")
    
    if not Path(task["bvh_path"]).exists():
        raise HTTPException(status_code=404, detail="Result file missing")
    
    return FileResponse(
        task["bvh_path"],
        filename=f"result_{task_id}.bvh",
        media_type="application/octet-stream"
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/", StaticFiles(directory="static", html=True), name="frontend")
