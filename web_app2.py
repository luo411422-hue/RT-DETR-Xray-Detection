# web_detection.py
import os, io, time, yaml, tempfile
import warnings
from collections import Counter

# 过滤警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import cv2
from PIL import Image
import streamlit as st

try:
    from ultralytics import YOLO
    # 如果需要其他版本
    # import yolov5
    # 其他导入...
except ImportError as e:
    st.error(f"❌ 模型导入失败: {e}")
    st.stop()

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="X射线违禁品检测系统",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 黑框白字CSS样式 ====================
st.markdown("""
<style>
    /* 主标题样式 */
    .main-title {
        text-align: center;
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        padding: 1rem;
        background: #000000;
        border-radius: 10px;
        border: 2px solid #444444;
    }
    
    /* 副标题样式 */
    .sub-title {
        color: #ffffff;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
        padding: 0.5rem;
        background: #111111;
        border-radius: 8px;
        border: 1px solid #333333;
    }
    
    /* 分区标题 */
    .section-header {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding: 0.8rem;
        background: #000000;
        border-radius: 8px;
        border: 2px solid #444444;
    }
    
    /* 卡片样式 - 黑框白字 */
    .card {
        background: #000000;
        color: #ffffff;
        border-radius: 10px;
        padding: 1.5rem;
        border: 2px solid #444444;
        margin: 1rem 0;
    }
    
    /* 文件上传区域 - 黑框白字 */
    .upload-section {
        background: #000000;
        color: #ffffff;
        border: 2px dashed #666666;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #ffffff;
        background: #111111;
    }
    
    /* 日志区域 - 黑框绿字 */
    .log-container {
        background: #000000;
        color: #00ff00;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        max-height: 400px;
        overflow-y: auto;
        border: 2px solid #333333;
    }
    
    /* 按钮样式 - 黑色背景 */
    .stButton button {
        background-color: #000000;
        color: #ffffff;
        border: 2px solid #444444;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #222222;
        border-color: #ffffff;
        transform: translateY(-1px);
    }
    
    /* 主要按钮样式 */
    .stButton button[kind="primary"] {
        background-color: #1a472a;
        border-color: #2e8b57;
    }
    
    .stButton button[kind="primary"]:hover {
        background-color: #2e8b57;
        border-color: #ffffff;
    }
    
    /* 侧边栏样式 - 黑色主题 */
    section[data-testid="stSidebar"] {
        background: #000000;
        color: #ffffff;
    }
    
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* 标签页样式 - 黑色主题 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #000000;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #111111;
        color: #ffffff;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        font-weight: 500;
        border: 1px solid #333333;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #222222;
        color: #ffffff;
        border-bottom: 2px solid #ffffff;
    }
    
    /* 统计信息样式 */
    .stats-box {
        background: #111111;
        color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #444444;
    }
    
    /* 输入框样式 - 黑色主题 */
    .stTextInput input {
        background: #000000;
        color: #ffffff;
        border: 2px solid #444444;
    }
    
    /* 滑块样式 */
    .stSlider * {
        color: #ffffff !important;
    }
    
    /* 复选框样式 */
    .stCheckbox * {
        color: #ffffff !important;
    }
    
    /* 选择框样式 */
    .stSelectbox * {
        color: #ffffff !important;
    }
    
    /* 整体页面背景 */
    .main {
        background: #111111;
    }
    
    /* 文件上传器内部文字 */
    .stFileUploader label {
        color: #ffffff !important;
    }
    
    /* 警告和错误消息 */
    .stAlert {
        background: #111111;
        color: #ffffff;
        border: 1px solid #444444;
    }
    
    /* 进度条 */
    .stProgress > div > div {
        background-color: #2e8b57;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 工具函数 ====================
def to_bgr_writeable(img_pil: Image.Image) -> np.ndarray:
    """PIL→BGR且可写可连续"""
    try:
        arr = np.array(img_pil.convert("RGB"))
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return np.ascontiguousarray(bgr).copy()
    except Exception as e:
        st.error(f"图像转换失败: {e}")
        return None

def analyse_counts(result, names):
    if result is None or len(result) == 0 or names is None: 
        return {}
    try:
        cls_ids = result[:, -1].astype(int) if not isinstance(result, list) else [int(x[-1]) for x in result]
        c = Counter(cls_ids)
        return {names[i]: int(c.get(i, 0)) for i in range(len(names)) if c.get(i, 0) > 0}
    except Exception:
        return {}

@st.cache_resource(show_spinner=False)
def init_model_by_yaml(yaml_path: str):
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        mt = cfg.get("model_type")
        if mt == "yolov5": 
            model = yolov5(**cfg)
        elif mt == "yolov7": 
            model = yolov7(**cfg)
        elif mt == "yolov8": 
            model = yolov8(**cfg)
        elif mt == "rtdetr": 
            model = rtdetr(**cfg)
        else: 
            raise ValueError(f"未知的 model_type: {mt}")
        return model
    except Exception as e:
        st.error(f"模型初始化失败: {e}")
        return None

def ensure_model_loaded():
    if "model" not in st.session_state or st.session_state["model"] is None:
        st.error("🚫 请先在左侧上传并加载 YAML 配置")
        return None
    return st.session_state["model"]

def save_image(pil_img: Image.Image, out_dir: str, filename: str):
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        pil_img.save(path)
        return path
    except Exception as e:
        st.error(f"保存图片失败: {e}")
        return None

# ==================== 侧边栏配置 ====================
with st.sidebar:
    st.markdown('<div class="main-title">🔍 违禁品检测系统</div>', unsafe_allow_html=True)
    
    # 模型配置区域
    st.markdown("### ⚙️ 模型配置")
    
    with st.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("**上传模型配置文件**")
        yaml_file = st.file_uploader(
            "选择YAML文件",
            type=["yaml"],
            label_visibility="collapsed",
            help="上传模型配置文件"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 按钮组
    col1, col2 = st.columns(2)
    with col1:
        load_btn = st.button("🔄 加载模型", use_container_width=True, type="primary")
    with col2:
        clear_log_btn = st.button("🗑️ 清空日志", use_container_width=True)
    
    # 保存目录
    save_dir = st.text_input("💾 结果保存目录", value="results")
    
    st.markdown("---")
    
    # 系统日志
    st.markdown("### 📋 系统日志")
    if "log" not in st.session_state:
        st.session_state["log"] = []
    
    # 日志显示区域
    st.markdown('<div class="log-container">', unsafe_allow_html=True)
    log_content = "\n".join(st.session_state["log"]) if st.session_state["log"] else "等待操作..."
    st.code(log_content, language="text")
    st.markdown('</div>', unsafe_allow_html=True)

# 日志函数
def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    st.session_state["log"].append(f"[{ts}] {msg}")

# 清空日志
if clear_log_btn:
    st.session_state["log"] = []
    st.rerun()

# 模型加载逻辑
if load_btn:
    if not yaml_file:
        st.sidebar.error("❌ 请先选择YAML配置文件")
    else:
        with st.spinner("🔄 模型加载中..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as tmp:
                tmp.write(yaml_file.read())
                tmp_path = tmp.name
            try:
                model = init_model_by_yaml(tmp_path)
                if model:
                    st.session_state["model"] = model
                    st.sidebar.success("✅ 模型加载成功")
                    log("模型加载成功")
                else:
                    st.sidebar.error("❌ 模型加载失败")
            except Exception as e:
                st.session_state["model"] = None
                st.sidebar.error(f"❌ 模型加载失败：{e}")
                log(f"模型加载失败: {e}")
        st.rerun()

# ==================== 主内容区域 ====================
st.markdown('<div class="main-title">基于深度学习的X射线违禁品检测系统</div>', unsafe_allow_html=True)

st.markdown("""
<div class="sub-title">
    使用先进的深度学习技术检测X射线图像中的违禁物品
</div>
""", unsafe_allow_html=True)

# 创建标签页
tab1, tab2, tab3, tab4 = st.tabs([
    "📸 单张图片检测", 
    "🖼️ 批量图片检测", 
    "🎥 视频文件检测", 
    "📹 实时摄像头"
])

# === 单张图片检测 ===
with tab1:
    st.markdown('<div class="section-header">📸 单张图片检测</div>', unsafe_allow_html=True)
    
    # 上传区域
    with st.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("**选择或拖拽图片文件**")
        uploaded_file = st.file_uploader(
            "上传图片",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="single_upload"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file:
        # 显示文件信息
        file_size = uploaded_file.size / 1024  # KB
        st.markdown(f'<div class="card">📄 已选择文件: <b>{uploaded_file.name}</b> ({file_size:.1f} KB)</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 显示原图
            st.image(uploaded_file, caption="📤 上传的图片", use_container_width=True)
            
            # 检测按钮
            if st.button("🚀 开始检测", type="primary", use_container_width=True):
                model = ensure_model_loaded()
                if model:
                    with st.spinner("🔍 检测中..."):
                        try:
                            img_pil = Image.open(uploaded_file).convert("RGB")
                            img_cv = to_bgr_writeable(img_pil)
                            
                            if img_cv is not None:
                                t0 = time.time()
                                det_bgr, result = model(img_cv)
                                dt = time.time() - t0
                                
                                out_pil = Image.fromarray(det_bgr[:, :, ::-1])
                                
                                # 结果显示
                                st.success(f"✅ 检测完成！用时 {dt:.3f} 秒")
                                
                                # 并排显示结果
                                result_col1, result_col2 = st.columns(2)
                                with result_col1:
                                    st.image(out_pil, caption="🟢 检测结果", use_container_width=True)
                                
                                with result_col2:
                                    # 统计信息
                                    names = getattr(model, "names", [])
                                    counts = analyse_counts(result, names)
                                    st.markdown('<div class="section-header">📊 检测统计</div>', unsafe_allow_html=True)
                                    
                                    if counts:
                                        for item, count in counts.items():
                                            st.markdown(f'<div class="stats-box">🔴 {item}: {count}</div>', unsafe_allow_html=True)
                                    else:
                                        st.markdown('<div class="card">📝 未检测到目标物体</div>', unsafe_allow_html=True)
                                    
                                    # 下载按钮
                                    buf = io.BytesIO()
                                    out_pil.save(buf, format="PNG")
                                    st.download_button(
                                        "💾 下载检测结果",
                                        buf.getvalue(),
                                        file_name=f"detected_{uploaded_file.name}",
                                        mime="image/png",
                                        use_container_width=True
                                    )
                                
                                # 保存结果
                                save_path = save_image(out_pil, save_dir, f"detected_{uploaded_file.name}")
                                if save_path:
                                    log(f"图片检测完成: {save_path}")
                        
                        except Exception as e:
                            st.error(f"❌ 检测失败: {e}")
                            log(f"检测失败: {e}")

# === 批量图片检测 ===
with tab2:
    st.markdown('<div class="section-header">🖼️ 批量图片检测</div>', unsafe_allow_html=True)
    
    # 批量上传区域
    with st.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("**选择多张图片进行批量检测**")
        uploaded_files = st.file_uploader(
            "批量上传图片",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="batch_upload"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_files:
        st.success(f"✅ 已选择 {len(uploaded_files)} 张图片")
        
        # 显示选择的文件列表
        with st.expander("📋 已选择的文件列表", expanded=True):
            for i, file in enumerate(uploaded_files):
                file_size = file.size / 1024
                st.markdown(f'<div class="card">{i+1}. <b>{file.name}</b> ({file_size:.1f} KB)</div>', unsafe_allow_html=True)
        
        # 处理选项
        col1, col2 = st.columns(2)
        with col1:
            batch_detect_btn = st.button("🚀 开始批量检测", type="primary", use_container_width=True)
        with col2:
            zip_option = st.checkbox("📦 完成后打包下载", value=True)
        
        if batch_detect_btn:
            model = ensure_model_loaded()
            if model:
                import zipfile
                
                # 创建进度指示器
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []
                
                # 处理每张图片
                for i, file in enumerate(uploaded_files):
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"🔄 处理中: {file.name} ({i+1}/{len(uploaded_files)})")
                    
                    try:
                        img_pil = Image.open(file).convert("RGB")
                        img_cv = to_bgr_writeable(img_pil)
                        
                        if img_cv is not None:
                            det_bgr, result = model(img_cv)
                            out_pil = Image.fromarray(det_bgr[:, :, ::-1])
                            
                            # 保存结果
                            save_path = save_image(out_pil, save_dir, f"batch_{file.name}")
                            
                            names = getattr(model, "names", [])
                            counts = analyse_counts(result, names)
                            
                            results.append({
                                "filename": file.name,
                                "counts": counts,
                                "path": save_path
                            })
                    
                    except Exception as e:
                        st.error(f"处理 {file.name} 时出错: {e}")
                        log(f"处理失败 {file.name}: {e}")
                
                # 清除进度指示器
                progress_bar.empty()
                status_text.empty()
                
                # 显示处理结果汇总
                st.markdown('<div class="section-header">📊 批量处理结果</div>', unsafe_allow_html=True)
                
                # 如果有ZIP选项，创建下载包
                if zip_option and results:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                        zip_path = tmp_zip.name
                    
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        for result in results:
                            if result['path'] and os.path.exists(result['path']):
                                zipf.write(result['path'], os.path.basename(result['path']))
                    
                    with open(zip_path, 'rb') as f:
                        st.download_button(
                            "📥 下载所有结果(ZIP)",
                            f.read(),
                            file_name="batch_results.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                
                # 显示详细结果
                for result in results:
                    with st.expander(f"📄 {result['filename']}", expanded=False):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if result['path'] and os.path.exists(result['path']):
                                st.image(result['path'], use_container_width=True)
                        with col_b:
                            if result['counts']:
                                st.markdown('<div class="card"><b>检测结果:</b></div>', unsafe_allow_html=True)
                                for item, count in result['counts'].items():
                                    st.markdown(f'<div class="card">- {item}: {count}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="card">📝 未检测到目标</div>', unsafe_allow_html=True)
                
                st.balloons()
                st.success(f"🎉 批量处理完成！共处理 {len(results)} 张图片")
                log(f"批量处理完成: {len(results)} 张图片")

# === 视频文件检测 ===
with tab3:
    st.markdown('<div class="section-header">🎥 视频文件检测</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("**上传视频文件**")
        video_file = st.file_uploader(
            "上传视频",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
            key="video_upload"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    if video_file:
        st.video(video_file)
        st.markdown(f'<div class="card">📹 已选择视频文件: <b>{video_file.name}</b></div>', unsafe_allow_html=True)
        
        # 视频处理选项
        col1, col2 = st.columns(2)
        with col1:
            frame_interval = st.slider("📊 抽帧间隔", 1, 10, 5, help="每隔N帧处理一帧")
        with col2:
            save_frames = st.checkbox("💾 保存处理帧", value=False)
        
        if st.button("🎬 开始视频分析", type="primary", use_container_width=True):
            model = ensure_model_loaded()
            if model:
                with st.spinner("🎥 处理视频中..."):
                    try:
                        # 保存临时视频文件
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                            tmp_video.write(video_file.read())
                            video_path = tmp_video.name
                        
                        # 处理视频
                        cap = cv2.VideoCapture(video_path)
                        frame_placeholder = st.empty()
                        frame_count = 0
                        processed_count = 0
                        start_time = time.time()
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            # 按间隔处理帧
                            if frame_count % frame_interval == 0:
                                det_bgr, _ = model(frame)
                                frame_rgb = det_bgr[:, :, ::-1]
                                frame_placeholder.image(frame_rgb, caption=f"帧 {frame_count}", use_container_width=True)
                                
                                if save_frames:
                                    save_path = os.path.join(save_dir, f"frame_{frame_count:06d}.png")
                                    Image.fromarray(frame_rgb).save(save_path)
                                
                                processed_count += 1
                            
                            frame_count += 1
                        
                        cap.release()
                        os.unlink(video_path)  # 删除临时文件
                        
                        processing_time = time.time() - start_time
                        st.success(f"✅ 视频分析完成！共处理 {processed_count} 帧，用时 {processing_time:.2f}秒")
                        log(f"视频分析完成: {processed_count} 帧")
                        
                    except Exception as e:
                        st.error(f"❌ 视频处理失败: {e}")
                        log(f"视频处理失败: {e}")

# === 实时摄像头 ===
with tab4:
    st.markdown('<div class="section-header">📹 实时摄像头检测</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    💡 <b>实时摄像头功能说明:</b><br>
    - 需要安装额外依赖: <code>pip install streamlit-webrtc av</code><br>
    - 需要在支持WebRTC的浏览器中使用<br>
    - 首次使用可能需要授权摄像头权限
    </div>
    """, unsafe_allow_html=True)
    
    camera_enabled = st.checkbox("启用摄像头检测", value=False)
    
    if camera_enabled:
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode
            import av
            
            model = ensure_model_loaded()
            if model:
                def video_frame_callback(frame):
                    img = frame.to_ndarray(format="bgr24")
                    det_bgr, _ = model(np.ascontiguousarray(img).copy())
                    return av.VideoFrame.from_ndarray(det_bgr, format="bgr24")
                
                webrtc_streamer(
                    key="xray-realtime-detection",
                    mode=WebRtcMode.SENDRECV,
                    video_frame_callback=video_frame_callback,
                    media_stream_constraints={"video": True, "audio": False},
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                )
        except ImportError:
            st.error("""
            <div class="card">
            ❌ <b>依赖未安装</b><br><br>
            请运行以下命令安装所需依赖:<br>
            <code>pip install streamlit-webrtc av</code>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ 摄像头启动失败：{e}")

# ==================== 页脚信息 ====================
st.markdown("---")
st.markdown(
    "<div class='card' style='text-align: center;'>"
    "🔍 X射线违禁品检测系统 | 基于深度学习技术 | 安全检测解决方案"
    "</div>", 
    unsafe_allow_html=True
)