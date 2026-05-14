from __future__ import annotations

import argparse
import ast
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DEFAULT_SPLITS = ("train", "valid", "test")
DEFAULT_AUGS = (
    "hflip",
    "vflip",
    "center_crop",
    "contrast",
    "gaussian_noise",
    "gaussian_blur",
    "sharpen",
)


@dataclass(frozen=True)
class Box:
    class_id: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class Sample:
    uid: str
    image_path: Path
    width: int | None
    height: int | None
    boxes: list[Box]
    class_ids: tuple[int, ...]
    parent_uid: str | None = None
    transform: dict | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a paper-style SIXray-s dataset from an existing YOLO-format SIXray dataset."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"e:\paper\transformer-master\SIXray"),
        help="Root directory of the current YOLO-format SIXray dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"e:\paper\transformer-master\SIXray-s"),
        help="Output directory for the generated SIXray-s dataset.",
    )
    parser.add_argument(
        "--base-size",
        type=int,
        default=4000,
        help="Number of original positive images sampled before offline augmentation.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=5000,
        help="Final number of images after augmentation.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--augmentations",
        nargs="+",
        default=list(DEFAULT_AUGS),
        choices=list(DEFAULT_AUGS),
        help="Offline augmentations used to grow the dataset.",
    )
    parser.add_argument(
        "--crop-ratio",
        type=float,
        default=0.8,
        help="Center crop keeps this fraction of width and height before resizing back.",
    )
    parser.add_argument(
        "--min-visible-ratio",
        type=float,
        default=0.3,
        help="Minimum retained box area ratio for the center crop augmentation.",
    )
    parser.add_argument(
        "--contrast-lower",
        type=float,
        default=0.5,
        help="Lower bound of random contrast gain.",
    )
    parser.add_argument(
        "--contrast-upper",
        type=float,
        default=1.5,
        help="Upper bound of random contrast gain.",
    )
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=12.0,
        help="Standard deviation used for Gaussian noise.",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=5,
        help="Odd kernel size used for Gaussian blur.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("random", "balanced"),
        default="balanced",
        help="How the initial positive images are sampled.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before writing the new dataset.",
    )
    return parser.parse_args()


def load_class_names(source_root: Path) -> list[str]:
    data_yaml = source_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Cannot find class config: {data_yaml}")

    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("names:"):
            continue
        names = ast.literal_eval(line.split(":", 1)[1].strip())
        if not isinstance(names, list) or not names:
            break
        return [str(name) for name in names]

    raise ValueError(f"Cannot parse class names from: {data_yaml}")


def read_image(image_path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return image


def write_image(image_path: Path, image: np.ndarray) -> None:
    suffix = image_path.suffix.lower() or ".jpg"
    ok, encoded = cv2.imencode(".jpg" if suffix == ".jpg" else suffix, image)
    if not ok:
        raise ValueError(f"Failed to encode image: {image_path}")
    encoded.tofile(str(image_path))


def get_image_size(image_path: Path) -> tuple[int, int]:
    suffix = image_path.suffix.lower()
    with image_path.open("rb") as fh:
        header = fh.read(32)

        if suffix == ".png":
            if header[:8] != b"\x89PNG\r\n\x1a\n":
                raise ValueError(f"Invalid PNG file: {image_path}")
            width = int.from_bytes(header[16:20], "big")
            height = int.from_bytes(header[20:24], "big")
            return width, height

        if suffix in {".jpg", ".jpeg"}:
            fh.seek(0)
            if fh.read(2) != b"\xff\xd8":
                raise ValueError(f"Invalid JPEG file: {image_path}")
            while True:
                marker_prefix = fh.read(1)
                if not marker_prefix:
                    break
                if marker_prefix != b"\xff":
                    continue
                marker = fh.read(1)
                while marker == b"\xff":
                    marker = fh.read(1)
                if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"}:
                    _ = int.from_bytes(fh.read(2), "big")
                    _precision = fh.read(1)
                    height = int.from_bytes(fh.read(2), "big")
                    width = int.from_bytes(fh.read(2), "big")
                    return width, height
                seg_len = int.from_bytes(fh.read(2), "big")
                fh.seek(seg_len - 2, 1)

        if suffix == ".bmp":
            fh.seek(18)
            width = int.from_bytes(fh.read(4), "little")
            height = int.from_bytes(fh.read(4), "little")
            return width, height

    image = read_image(image_path)
    height, width = image.shape[:2]
    return width, height


def load_positive_samples(source_root: Path) -> tuple[list[Sample], Counter, Counter]:
    class_names = load_class_names(source_root)
    image_level_counts: Counter = Counter()
    box_level_counts: Counter = Counter()
    samples: list[Sample] = []

    for split in DEFAULT_SPLITS:
        image_dir = source_root / split / "images"
        label_dir = source_root / split / "labels"
        if not image_dir.exists() or not label_dir.exists():
            continue

        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue

            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                continue

            boxes: list[Box] = []
            class_ids: set[int] = set()

            for raw_line in text.splitlines():
                parts = raw_line.split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
                xmin = xc - bw / 2.0
                ymin = yc - bh / 2.0
                xmax = xc + bw / 2.0
                ymax = yc + bh / 2.0
                if xmax <= xmin or ymax <= ymin:
                    continue
                boxes.append(Box(class_id, xmin, ymin, xmax, ymax))
                class_ids.add(class_id)
                box_level_counts[class_names[class_id]] += 1

            if not boxes:
                continue

            for class_id in class_ids:
                image_level_counts[class_names[class_id]] += 1

            samples.append(
                Sample(
                    uid=image_path.stem,
                    image_path=image_path,
                    width=None,
                    height=None,
                    boxes=boxes,
                    class_ids=tuple(sorted(class_ids)),
                )
            )

    if not samples:
        raise ValueError(f"No positive samples found under: {source_root}")
    return samples, image_level_counts, box_level_counts


def ensure_clean_output(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_root}. Use --overwrite to replace it.")
        for path in sorted(output_root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            else:
                path.rmdir()
        output_root.rmdir()

    for split in DEFAULT_SPLITS:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)


def sample_initial_records(
    samples: list[Sample],
    base_size: int,
    selection_mode: str,
    rng: random.Random,
) -> list[Sample]:
    if base_size <= 0 or base_size >= len(samples):
        picked = list(samples)
        rng.shuffle(picked)
        return picked

    pool = list(samples)
    rng.shuffle(pool)

    if selection_mode == "random":
        return pool[:base_size]

    selected: list[Sample] = []
    selected_ids: set[str] = set()
    image_counts = Counter()

    while len(selected) < base_size and pool:
        pool.sort(key=lambda sample: sum(image_counts[class_id] for class_id in sample.class_ids))
        sample = pool.pop(0)
        if sample.uid in selected_ids:
            continue
        selected.append(sample)
        selected_ids.add(sample.uid)
        for class_id in sample.class_ids:
            image_counts[class_id] += 1

    return selected


def hydrate_sample(sample: Sample) -> Sample:
    
    if sample.width is not None and sample.height is not None:
        return sample

    width, height = get_image_size(sample.image_path)
    boxes = [
        Box(
            box.class_id,
            max(0.0, min(box.xmin * width, width - 1)),
            max(0.0, min(box.ymin * height, height - 1)),
            max(0.0, min(box.xmax * width, width - 1)),
            max(0.0, min(box.ymax * height, height - 1)),
        )
        for box in sample.boxes
    ]
    boxes = [box for box in boxes if box.xmax > box.xmin and box.ymax > box.ymin]
    if not boxes:
        raise ValueError(f"Sample has no valid boxes after hydration: {sample.image_path}")

    return Sample(
        uid=sample.uid,
        image_path=sample.image_path,
        width=width,
        height=height,
        boxes=boxes,
        class_ids=sample.class_ids,
        parent_uid=sample.parent_uid,
        transform=sample.transform,
    )


def to_class_name_counts(samples: Iterable[Sample], class_names: list[str]) -> Counter:
    counts: Counter = Counter()
    for sample in samples:
        for class_id in sample.class_ids:
            counts[class_names[class_id]] += 1
    return counts


def make_transform(op_name: str, args: argparse.Namespace, rng: random.Random) -> dict:
    if op_name == "center_crop":
        return {"name": op_name, "crop_ratio": args.crop_ratio, "min_visible_ratio": args.min_visible_ratio}
    if op_name == "contrast":
        alpha = rng.uniform(args.contrast_lower, args.contrast_upper)
        return {"name": op_name, "alpha": alpha}
    if op_name == "gaussian_noise":
        return {"name": op_name, "sigma": args.noise_sigma}
    if op_name == "gaussian_blur":
        kernel = args.blur_kernel if args.blur_kernel % 2 == 1 else args.blur_kernel + 1
        return {"name": op_name, "kernel": max(3, kernel)}
    if op_name == "sharpen":
        return {"name": op_name}
    if op_name in {"hflip", "vflip"}:
        return {"name": op_name}
    raise ValueError(f"Unsupported augmentation: {op_name}")


def transform_boxes(boxes: list[Box], width: int, height: int, transform: dict) -> list[Box]:
    name = transform["name"]
    if name == "hflip":
        return [Box(box.class_id, width - box.xmax, box.ymin, width - box.xmin, box.ymax) for box in boxes]
    if name == "vflip":
        return [Box(box.class_id, box.xmin, height - box.ymax, box.xmax, height - box.ymin) for box in boxes]
    if name != "center_crop":
        return list(boxes)

    crop_ratio = float(transform["crop_ratio"])
    min_visible_ratio = float(transform["min_visible_ratio"])
    crop_w = max(2, int(round(width * crop_ratio)))
    crop_h = max(2, int(round(height * crop_ratio)))
    x0 = max(0, (width - crop_w) // 2)
    y0 = max(0, (height - crop_h) // 2)
    scale_x = width / crop_w
    scale_y = height / crop_h

    transformed: list[Box] = []
    for box in boxes:
        ixmin = max(box.xmin, x0)
        iymin = max(box.ymin, y0)
        ixmax = min(box.xmax, x0 + crop_w)
        iymax = min(box.ymax, y0 + crop_h)
        if ixmax <= ixmin or iymax <= iymin:
            continue

        old_area = max(1.0, (box.xmax - box.xmin) * (box.ymax - box.ymin))
        new_area = (ixmax - ixmin) * (iymax - iymin)
        if new_area / old_area < min_visible_ratio:
            continue

        nxmin = (ixmin - x0) * scale_x
        nymin = (iymin - y0) * scale_y
        nxmax = (ixmax - x0) * scale_x
        nymax = (iymax - y0) * scale_y
        transformed.append(Box(box.class_id, nxmin, nymin, nxmax, nymax))

    return transformed


def apply_transform(image: np.ndarray, transform: dict) -> np.ndarray:
    name = transform["name"]
    if name == "hflip":
        return cv2.flip(image, 1)
    if name == "vflip":
        return cv2.flip(image, 0)
    if name == "center_crop":
        crop_ratio = float(transform["crop_ratio"])
        height, width = image.shape[:2]
        crop_w = max(2, int(round(width * crop_ratio)))
        crop_h = max(2, int(round(height * crop_ratio)))
        x0 = max(0, (width - crop_w) // 2)
        y0 = max(0, (height - crop_h) // 2)
        cropped = image[y0:y0 + crop_h, x0:x0 + crop_w]
        return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
    if name == "contrast":
        alpha = float(transform["alpha"])
        return cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    if name == "gaussian_noise":
        sigma = float(transform["sigma"])
        noise = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
        noised = image.astype(np.float32) + noise
        return np.clip(noised, 0, 255).astype(np.uint8)
    if name == "gaussian_blur":
        kernel = int(transform["kernel"])
        return cv2.GaussianBlur(image, (kernel, kernel), 0)
    if name == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(image, -1, kernel)
    raise ValueError(f"Unsupported transform: {name}")


def build_augmented_pool(
    base_samples: list[Sample],
    target_size: int,
    augmentations: list[str],
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[list[Sample], Counter]:
    hydrated_base = [hydrate_sample(sample) for sample in base_samples]

    if len(hydrated_base) >= target_size:
        return hydrated_base[:target_size], Counter()

    pool = list(hydrated_base)
    op_counts: Counter = Counter()
    class_counts = Counter()
    for sample in pool:
        for class_id in sample.class_ids:
            class_counts[class_id] += 1

    attempts = 0
    while len(pool) < target_size:
        attempts += 1
        if attempts > (target_size - len(base_samples)) * 20:
            raise RuntimeError("Failed to create enough valid augmentations. Try a larger base set or milder crop.")

        weights = []
        for sample in pool:
            score = 0.0
            for class_id in sample.class_ids:
                score += 1.0 / max(class_counts[class_id], 1)
            weights.append(score or 1.0)

        parent = rng.choices(pool, weights=weights, k=1)[0]
        op_name = rng.choice(augmentations)
        transform = make_transform(op_name, args, rng)
        new_boxes = transform_boxes(parent.boxes, parent.width, parent.height, transform)
        if not new_boxes:
            continue

        class_ids = tuple(sorted({box.class_id for box in new_boxes}))
        sample = Sample(
            uid=f"{parent.uid}_{op_name}_{len(pool):05d}",
            image_path=parent.image_path,
            width=parent.width,
            height=parent.height,
            boxes=new_boxes,
            class_ids=class_ids,
            parent_uid=parent.uid,
            transform=transform,
        )
        pool.append(sample)
        op_counts[op_name] += 1
        for class_id in class_ids:
            class_counts[class_id] += 1

    return pool, op_counts


def materialize_image(sample: Sample, sample_map: dict[str, Sample]) -> np.ndarray:
    if sample.parent_uid is None or sample.transform is None:
        return read_image(sample.image_path)
    parent_image = materialize_image(sample_map[sample.parent_uid], sample_map)
    return apply_transform(parent_image, sample.transform)


def split_samples(samples: list[Sample], args: argparse.Namespace, rng: random.Random) -> dict[str, list[Sample]]:
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    items = list(samples)
    rng.shuffle(items)
    train_end = int(len(items) * args.train_ratio)
    val_end = train_end + int(len(items) * args.val_ratio)
    splits = {
        "train": items[:train_end],
        "valid": items[train_end:val_end],
        "test": items[val_end:],
    }
    if not all(splits.values()):
        raise ValueError("One split is empty. Adjust ratios or target size.")
    return splits


def write_label(label_path: Path, boxes: list[Box], width: int, height: int) -> None:
    lines = []
    for box in boxes:
        xc = ((box.xmin + box.xmax) / 2.0) / width
        yc = ((box.ymin + box.ymax) / 2.0) / height
        bw = (box.xmax - box.xmin) / width
        bh = (box.ymax - box.ymin) / height
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"{box.class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def export_dataset(
    output_root: Path,
    splits: dict[str, list[Sample]],
    class_names: list[str],
) -> dict[str, dict[str, object]]:
    sample_map = {sample.uid: sample for records in splits.values() for sample in records}
    image_cache: dict[str, np.ndarray] = {}
    stats: dict[str, dict[str, object]] = {}

    def materialize_cached(sample: Sample) -> np.ndarray:
        cached = image_cache.get(sample.uid)
        if cached is not None:
            return cached.copy()
        image = materialize_image(sample, sample_map)
        image_cache[sample.uid] = image
        return image.copy()

    for split_name, records in splits.items():
        image_counts = Counter()
        box_counts = Counter()
        image_dir = output_root / split_name / "images"
        label_dir = output_root / split_name / "labels"

        for sample in records:
            image = materialize_cached(sample)
            image_out = image_dir / f"{sample.uid}.jpg"
            label_out = label_dir / f"{sample.uid}.txt"
            write_image(image_out, image)
            write_label(label_out, sample.boxes, sample.width, sample.height)

            for class_id in sample.class_ids:
                image_counts[class_names[class_id]] += 1
            for box in sample.boxes:
                box_counts[class_names[box.class_id]] += 1

        stats[split_name] = {
            "images": len(records),
            "image_class_counts": dict(image_counts),
            "box_class_counts": dict(box_counts),
        }

    yaml_lines = [
        f"path: {output_root.resolve().as_posix()}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        f"nc: {len(class_names)}",
        f"names: {class_names}",
    ]
    (output_root / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    class_names = load_class_names(args.source_root)
    all_samples, all_image_counts, all_box_counts = load_positive_samples(args.source_root)
    ensure_clean_output(args.output_root, args.overwrite)

    base_samples = sample_initial_records(all_samples, args.base_size, args.selection_mode, rng)
    final_samples, op_counts = build_augmented_pool(base_samples, args.target_size, args.augmentations, args, rng)
    splits = split_samples(final_samples, args, rng)
    split_stats = export_dataset(args.output_root, splits, class_names)

    report = {
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "class_names": class_names,
        "source_positive_images": len(all_samples),
        "source_image_class_counts": dict(all_image_counts),
        "source_box_class_counts": dict(all_box_counts),
        "base_size": len(base_samples),
        "target_size": len(final_samples),
        "selection_mode": args.selection_mode,
        "augmentations": list(args.augmentations),
        "offline_augmentation_counts": dict(op_counts),
        "splits": split_stats,
        "notes": [
            "This script rebuilds a paper-style SIXray-s dataset from the current YOLO-format SIXray export.",
            "The paper text says six augmentations, but the listed operations are seven if horizontal and vertical flips are counted separately.",
            "This implementation keeps the paper-style order of augmenting first and splitting later.",
        ],
    }
    (args.output_root / "stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("SIXray-s dataset generation completed.")
    print(f"Source positive images: {len(all_samples)}")
    print(f"Base images selected: {len(base_samples)}")
    print(f"Final dataset size: {len(final_samples)}")
    print(f"Output dataset: {args.output_root}")
    print(f"Training yaml: {args.output_root / 'data.yaml'}")
    print(f"Stats report: {args.output_root / 'stats.json'}")


if __name__ == "__main__":
    main()
