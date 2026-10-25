
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch



from ultralytics import RTDETR, YOLO
from ultralytics.utils import yaml_load
from ultralytics.utils.torch_utils import get_flops, get_num_params, select_device

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_CONFIG = Path(r"pyqt-master\experiment.yaml")


from typing import Optional, Union


def normalize_string_literal(value):
    """“规范化字符串字面量”"""
    if not isinstance(value, str):
        return value

    text = value.strip()
    if len(text) >= 3 and text[0] in {"r", "R"} and text[1] in {'"', "'"} and text[-1] == text[1]:
        return text[2:-1]
    if len(text) >= 2 and text[0] in {'"', "'"} and text[-1] == text[0]:
        return text[1:-1]
    return text  #这个函数就像是一个安全过滤器，让你能够兼容各种“不小心多加了引号”的字符串输入，使脚本更鲁棒。


def normalize_config(data: dict) -> dict:
    """作用：递归地“清洗”整个配置字典的所有字符串值。
它利用 normalize_string_literal 去除每个值可能包含的多余引号或原始字符串前缀。"""
    return {key: normalize_string_literal(value) for key, value in data.items()}


def load_config(config_path: Optional[Union[str, Path]] = None) -> SimpleNamespace:
    """作用：加载 YAML 配置文件，清洗内容，并返回一个可点号访问的命名空间对象。"""
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path.resolve()}")
    data = normalize_config(yaml_load(path))
    data["config_path"] = str(path.resolve())
    return SimpleNamespace(**data)


def build_model(model_path: str):
    """这个 build_model 函数是一个模型加载分发器，
    它根据你提供的模型文件路径，自动判断应该用 RTDETR 还是 YOLO 来加载，并返回对应的模型对象"""
    suffix = Path(model_path).suffix.lower() # 获取文件后缀     
    if "rtdetr" in Path(model_path).stem.lower():
        return RTDETR(model_path)
    if suffix in {".yaml", ".yml"}:
        return YOLO(model_path)
    if suffix == ".pt":
        try:
            return RTDETR(model_path)
        except Exception:
            return YOLO(model_path)
    return YOLO(model_path)


def print_header(cfg: SimpleNamespace):
    """print_header 函数的作用是在实验开始前，用 JSON 格式打印出关键的运行配置信息，
    方便你在查看日志时快速确认本次实验跑的是什么模型、用的哪个数据集、配置来自哪里。"""
    print(json.dumps({ #打印一个JSON格式的字符串，包含了实验的配置信息
        "mode": cfg.mode,
        "model": cfg.model,
        "data": getattr(cfg, "data", None),
        "config_path": cfg.config_path,
    }, indent=2, ensure_ascii=False))


def get_run_dir(cfg: SimpleNamespace) -> Path:
    project = Path(getattr(cfg, "project", "runs/train"))
    name = getattr(cfg, "name", "exp")
    return project / name


def resolve_trained_weight_path(cfg: SimpleNamespace) -> Path:
    weights_dir = get_run_dir(cfg) / "weights"
    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last.pt"
    if best_path.exists():
        return best_path
    if last_path.exists():
        return last_path
    raise FileNotFoundError(
        f"No trained weights found under: {weights_dir.resolve()}\n"
        "Expected best.pt or last.pt after training."
    )


def train_model(cfg: SimpleNamespace):
    """train_model 是实验脚本中负责模型训练的核心函数。它完全由配置文件驱动，
    不需要你手动在代码里改参数。。"""
    model = build_model(cfg.model) #   根据配置文件中的模型路径，加载对应的模型对象（RTDETR 或 YOLO）
    metrics = model.train(
        data=cfg.data,
        imgsz=getattr(cfg, "imgsz", 640),
        epochs=getattr(cfg, "epochs", 100),
        batch=getattr(cfg, "batch", 16),
        workers=getattr(cfg, "workers", 8),
        device=getattr(cfg, "device", "0"),
        optimizer=getattr(cfg, "optimizer", "AdamW"),
        lr0=getattr(cfg, "lr0", 1e-4),
        project=getattr(cfg, "project", "runs/train"),
        name=getattr(cfg, "name", "exp"),
        pretrained=getattr(cfg, "pretrained", False),
    )
    trained_weight_path = resolve_trained_weight_path(cfg)
    print(f"trained_weights: {trained_weight_path.resolve()}")
    print("train_metrics:")
    print(metrics)
    return trained_weight_path


def val_model(cfg: SimpleNamespace, model_path: Optional[Union[str, Path]] = None):
    """val_model 是实验脚本中的验证/评估专用函数，它完全由配置对象 cfg 驱动，对训练好的模型进行精度和性能评估"""
    weight_path = Path(model_path) if model_path else Path(cfg.model)
    print(f"val_model_path: {weight_path}")
    model = build_model(str(weight_path))
    metrics = model.val(
        data=cfg.data,
        imgsz=getattr(cfg, "imgsz", 640),
        batch=getattr(cfg, "batch", 16),
        workers=getattr(cfg, "workers", 8),
        device=getattr(cfg, "device", "0"),
        split=getattr(cfg, "split", "val"),
        save_json=getattr(cfg, "save_json", False),
        plots=getattr(cfg, "plots", False),
    )

    results = {
        "precision": metrics.results_dict.get("metrics/precision(B)"),
        "recall": metrics.results_dict.get("metrics/recall(B)"),
        "mAP50": metrics.results_dict.get("metrics/mAP50(B)"),
        "mAP50_95": metrics.results_dict.get("metrics/mAP50-95(B)"),
        "speed_ms": metrics.speed,
    } #在未来的批量对比脚本中，val_model 就是精确收割这些指标的函数。每次验证完一个模型，把它的 mAP50、mAP50-95、速度等收集起来，最后汇总成 CSV。
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return results


def benchmark_fps(cfg: SimpleNamespace, model_path: Optional[Union[str, Path]] = None):
    """benchmark_fps 是实验脚本中专门用来测量模型推理速度（FPS）和计算/参数量的函数。
    它模拟单张图片在 GPU/CPU 上反复推理的平均耗时，并与参数量、GFLOPs 一起输出"""
    weight_path = Path(model_path) if model_path else Path(cfg.model)
    print(f"fps_model_path: {weight_path}")
    model = build_model(str(weight_path))
    device = select_device(getattr(cfg, "device", "0"), verbose=False)
    model.to(device)
    model.fuse()

    imgsz = getattr(cfg, "imgsz", 640)
    warmup = max(getattr(cfg, "warmup", 30), 1)
    runs = max(getattr(cfg, "runs", 200), 1)
    shape = (1, 3, imgsz, imgsz)
    x = torch.randn(shape, device=device)

    for _ in range(warmup):
        _ = model.model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(runs):
        _ = model.model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    avg_ms = elapsed / runs * 1000
    fps = runs / elapsed
    flops = get_flops(model.model, imgsz)
    params = get_num_params(model.model)

    results = {
        "model": cfg.model,
        "imgsz": imgsz,
        "device": str(device),
        "avg_inference_ms": avg_ms,
        "fps": fps,
        "gflops": flops,
        "params": params,
    }
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return results


def profile_model(cfg: SimpleNamespace, model_path: Optional[Union[str, Path]] = None):
    """实验脚本中专门用来剖析模型结构细节和计算量的函数。它会打印模型每一层的名称、参数量、计算量（GFLOPs）、特征图尺寸等详细信息，同时把汇总结果整理成 JSON 输出。"""
    weight_path = Path(model_path) if model_path else Path(cfg.model)
    print(f"profile_model_path: {weight_path}")
    model = build_model(str(weight_path))
    device = select_device(getattr(cfg, "device", "0"), verbose=False)
    model.to(device)
    model.fuse()
    layers, params, gradients, flops = model.model.info(verbose=True, imgsz=getattr(cfg, "imgsz", 640))
    results = {
        "layers": layers,
        "params": params,
        "gradients": gradients,
        "gflops": flops,
    }
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return results

def full_benchmark(cfg):
    print_header(cfg)
    trained_weight_path = train_model(cfg)
    val_reults = val_model(cfg, trained_weight_path)
    fps_results = benchmark_fps(cfg, trained_weight_path)
    profile_results = profile_model(cfg, trained_weight_path)

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = load_config(config_path)
    #print_header(cfg)
    # mode = getattr(cfg, "mode", "").lower()
    # if mode == "train":
    #     train_model(cfg)
    # elif mode == "val":
    #     val_model(cfg)
    # elif mode == "fps":
    #     benchmark_fps(cfg)
    # elif mode == "profile":
    #     profile_model(cfg)
    # else:
    #     raise ValueError("mode must be one of: train, val, fps, profile")

    full_benchmark(cfg)

if __name__ == "__main__":
    main()

