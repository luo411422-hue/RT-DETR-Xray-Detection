# web_detection.py  /  app.py
import os, io, time, yaml, tempfile
from collections import Counter

import numpy as np
import cv2
from PIL import Image
import streamlit as st

# 你的模型封装（保持与 yolo.py 一致）
from yolo import yolov5, yolov7, yolov8, rtdetr

# ----------------- 页面设置 & 小字号 -----------------
st.set_page_config(page_title="X射线违禁品检测（Web）", layout="wide")
st.markdown("""
<style>
html, body, [class*="css"] { font-size: 14px !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] label p { font-size: 14px !important; }
</style>
""", unsafe_allow_html=True)

# ----------------- 工具函数 -----------------
def to_bgr_writeable(img_pil: Image.Image) -> np.ndarray:
    """PIL→BGR且可写可连续，供OpenCV画框与模型推理"""
    arr = np.array(img_pil.convert("RGB"))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(bgr).copy()

def analyse_counts(result, names):
    if result is None or len(result) == 0 or names is None: return {}
    cls_ids = result[:, -1].astype(int) if not isinstance(result, list) else [int(x[-1]) for x in result]
    c = Counter(cls_ids)
    return {names[i]: int(c.get(i, 0)) for i in range(len(names)) if c.get(i, 0) > 0}

@st.cache_resource(show_spinner=False)
def init_model_by_yaml(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mt = cfg.get("model_type")
    if mt == "yolov5": model = yolov5(**cfg)
    elif mt == "yolov7": model = yolov7(**cfg)
    elif mt == "yolov8": model = yolov8(**cfg)
    elif mt == "rtdetr": model = rtdetr(**cfg)
    else: raise ValueError(f"未知的 model_type: {mt}")
    return model

def ensure_model_loaded():
    if "model" not in st.session_state or st.session_state["model"] is None:
        st.error("请先在左侧上传并加载 YAML 配置")
        return None
    return st.session_state["model"]

def save_image(pil_img: Image.Image, out_dir: str, filename: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    pil_img.save(path)
    return path

# ----------------- 侧边栏：加载模型 / 保存目录 -----------------
st.sidebar.subheader("模型与参数")
yaml_file = st.sidebar.file_uploader("上传模型配置（.yaml）", type=["yaml"])
col1, col2 = st.sidebar.columns(2)
with col1:
    load_btn = st.button("加载/重载模型", use_container_width=True)
with col2:
    clear_log = st.button("清空日志", use_container_width=True)
save_dir = st.sidebar.text_input("结果保存目录", value="result")

if clear_log: st.session_state["log"] = []
if "log" not in st.session_state: st.session_state["log"] = []

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    st.session_state["log"].append(f"[{ts}] {msg}")

if load_btn:
    if not yaml_file:
        st.sidebar.error("请选择 .yaml 配置文件")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as tmp:
            tmp.write(yaml_file.read())
            tmp_path = tmp.name
        try:
            st.session_state["model"] = init_model_by_yaml(tmp_path)
            st.sidebar.success("模型加载成功")
            log("load yaml success.")
        except Exception as e:
            st.session_state["model"] = None
            st.sidebar.error(f"模型加载失败：{e}")
            log(f"load yaml failed: {e}")

# ----------------- 主体：四个 Tab -----------------
st.title("基于深度学习的X射线违禁品检测系统（Web）")
tab_img, tab_batch, tab_video, tab_cam = st.tabs(["单张图片", "批量图片（多选）", "视频文件", "摄像头"])

# === 单张图片 ===
with tab_img:
    left, right = st.columns([2.2, 1])
    with left:
        up = st.file_uploader("上传图片（jpg/png）", type=["jpg","jpeg","png"])
        if up is not None:
            st.image(up, caption="输入图像", use_container_width=True)
        if st.button("开始检测", type="primary"):
            model = ensure_model_loaded()
            if model and up:
                img_pil = Image.open(up).convert("RGB")
                img_cv = to_bgr_writeable(img_pil)
                t0 = time.time()
                det_bgr, result = model(img_cv)  # 你的 yolo 类返回 (画好框的BGR, result)
                dt = time.time() - t0
                out_pil = Image.fromarray(det_bgr[:, :, ::-1])
                st.success(f"完成，用时 {dt:.3f}s")
                st.image(out_pil, caption="检测结果", use_column_width=True)

                names = getattr(model, "names", [])
                st.subheader("统计")
                st.json(analyse_counts(result, names) or {"提示":"未检测到目标"})

                buf = io.BytesIO(); out_pil.save(buf, format="PNG")
                st.download_button("下载结果图", buf.getvalue(), file_name="result.png", mime="image/png")
                p = save_image(out_pil, save_dir, "result.png")
                log(f"save image in {p}")
    with right:
        st.markdown("**日志**")
        st.code("\n".join(st.session_state["log"]) if st.session_state["log"] else "—")

# === 批量图片 ===
with tab_batch:
    up_list = st.file_uploader("一次选择多张图片", type=["jpg","jpeg","png"], accept_multiple_files=True)
    colb1, colb2 = st.columns([1,1])
    with colb1:
        run_batch = st.button("批量检测", type="primary")
    with colb2:
        zip_download = st.checkbox("完成后打包ZIP下载", value=True)
    ph_prog = st.empty()
    ph_grid = st.container()

    if run_batch:
        model = ensure_model_loaded()
        if model and up_list:
            import zipfile
            tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name if zip_download else None
            if tmp_zip:
                zf = zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED)
            results_summary = []
            for i, f in enumerate(up_list, start=1):
                ph_prog.progress(i/len(up_list), text=f"处理 {f.name} ({i}/{len(up_list)})")
                img_pil = Image.open(f).convert("RGB")
                img_cv = to_bgr_writeable(img_pil)
                det_bgr, result = model(img_cv)
                out_pil = Image.fromarray(det_bgr[:, :, ::-1])
                counts = analyse_counts(result, getattr(model, "names", []))
                save_path = save_image(out_pil, save_dir, f"{os.path.splitext(f.name)[0]}_det.png")
                results_summary.append((f.name, counts, save_path))
                if zip_download:
                    buf = io.BytesIO(); out_pil.save(buf, format="PNG")
                    zf.writestr(f"{os.path.splitext(f.name)[0]}_det.png", buf.getvalue())
            if zip_download:
                zf.close()
                with open(tmp_zip, "rb") as fh:
                    st.download_button("下载全部ZIP", fh.read(), file_name="batch_results.zip")
            with ph_grid:
                for name, counts, path in results_summary:
                    st.markdown(f"**{name}**  →  {counts if counts else '无检测'}  | 保存: `{path}`")
                    st.image(path, use_column_width=True)
            st.success(f"批量完成，共 {len(up_list)} 张")
            log(f"batch save dir: {os.path.abspath(save_dir)}")

# === 视频文件 ===
with tab_video:
    vfile = st.file_uploader("上传视频（mp4/avi）", type=["mp4","avi","mov","mkv"])
    run_video = st.button("开始解析视频", type="primary")
    frame_step = st.slider("抽帧间隔（每隔N帧处理一次）", 1, 10, 2)
    save_every = st.checkbox("保存每一帧结果到目录", value=False)

    if run_video:
        model = ensure_model_loaded()
        if model and vfile:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=f".{vfile.name.split('.')[-1]}")
            tf.write(vfile.read()); tf.flush()
            cap = cv2.VideoCapture(tf.name)
            ph_img = st.empty()
            ph_txt = st.empty()
            i = 0; t_all = time.time()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                if i % frame_step == 0:
                    det_bgr, _ = model(np.ascontiguousarray(frame).copy())
                    out_rgb = det_bgr[:, :, ::-1]
                    ph_img.image(out_rgb, caption=f"帧 {i}", use_column_width=True)
                    if save_every:
                        Image.fromarray(out_rgb).save(os.path.join(save_dir, f"frame_{i:06d}.png"))
                i += 1
            cap.release()
            st.success(f"视频处理完成，共 {i} 帧，用时 {time.time()-t_all:.2f}s")
            log(f"video processed, frames={i}")

# === 摄像头（WebRTC） ===
with tab_cam:
    st.markdown("需要先安装依赖：`pip install streamlit-webrtc av`")
    enable = st.checkbox("启动摄像头实时检测", value=False)
    if enable:
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode
            import av  # noqa

            model = ensure_model_loaded()
            if model:
                def video_frame_callback(frame):
                    img = frame.to_ndarray(format="bgr24")
                    det_bgr, _ = model(np.ascontiguousarray(img).copy())
                    return av.VideoFrame.from_ndarray(det_bgr, format="bgr24")

                webrtc_streamer(
                    key="xray-detect",
                    mode=WebRtcMode.SENDRECV,
                    video_frame_callback=video_frame_callback,
                    media_stream_constraints={"video": True, "audio": False},
                )
        except Exception as e:
            st.error(f"摄像头功能需要 streamlit-webrtc / av 支持：{e}")
