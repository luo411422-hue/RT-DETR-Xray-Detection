from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, cv2, yaml, io, time, os
import numpy as np
from threading import Event
from inference import ModelHub


app = FastAPI(title="Xray Detector Web API")
app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)


stop_flag = Event()


@app.post("/api/load_config")
async def load_config(file: UploadFile = File(...)):
    """加载 YAML 配置并初始化模型。"""
    cfg = yaml.safe_load((await file.read()).decode("utf-8"))
    ModelHub.load_from_cfg(cfg) # 内部完成权重加载/阈值设置等
    return {"ok": True, "msg": f"Loaded: {file.filename}"}


@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...)):
    """单图检测，返回标注后的 PNG。"""
    if not ModelHub.ready():
        return JSONResponse({"ok": False, "msg": "Model not ready"}, status_code=400)
    data = await file.read()
    arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    vis = ModelHub.infer_image(arr) # 返回可视化后的 BGR 图
    _, buf = cv2.imencode(".png", vis)
    return Response(content=buf.tobytes(), media_type="image/png")


@app.get("/api/stream")
async def stream(src: str = "0"):
    """视频/摄像头流检测（MJPEG）。src=文件路径 或 摄像头索引（字符串，如"0"）。"""
    if not ModelHub.ready():
        return JSONResponse({"ok": False, "msg": "Model not ready"}, status_code=400)
    stop_flag.clear()


# 解析 src：数字字符串视为摄像头索引
    cap = None
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return JSONResponse({"ok": False, "msg": f"Cannot open: {src}"}, status_code=400)


    def gen():
        while not stop_flag.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            frame = ModelHub.infer_stream(frame)
            _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            chunk = (b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            yield chunk
            time.sleep(0.01)
            cap.release()
        return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')


@app.post("/api/stop")
async def stop():
    stop_flag.set()
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)