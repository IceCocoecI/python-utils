import logging
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
import torchvision.ops as ops
from groundingdino.util.inference import load_model, load_image, predict
from segment_anything import sam_model_registry, SamPredictor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ASSETS_DIR = Path("/home/cz/software/pycharm-2024.2.3/PycharmProjects/python-utils/assets/grounding_dino_sam")
GROUNDING_DINO_CONFIG = ASSETS_DIR / "GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = ASSETS_DIR / "groundingdino_swint_ogc.pth"
SAM_CHECKPOINT = ASSETS_DIR / "sam_vit_h_4b8939.pth"
SAM_ENCODER_VERSION = "vit_h"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info("Using device: %s", DEVICE)
logger.info("Loading GroundingDINO from %s", GROUNDING_DINO_CHECKPOINT)
grounding_dino_model = load_model(str(GROUNDING_DINO_CONFIG), str(GROUNDING_DINO_CHECKPOINT))
# 关键修复：将 GroundingDINO 模型移到 GPU
grounding_dino_model = grounding_dino_model.to(DEVICE)

logger.info("Loading SAM (%s) from %s", SAM_ENCODER_VERSION, SAM_CHECKPOINT)
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=str(SAM_CHECKPOINT)).to(device=DEVICE)
sam_predictor = SamPredictor(sam)


def normalize_prompt(text_prompt: str) -> str:
    """
    规范化文本提示格式：
    - 转小写
    - 用 ". " 分隔多个类别
    - 确保以 "." 结尾
    """
    # 支持多种分隔符: ",", ";", ".", " "
    import re
    parts = re.split(r'[,;.\s]+', text_prompt.lower())
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return ""
    # GroundingDINO 期望格式: "person . sheep ." 或 "person. sheep."
    return " . ".join(parts) + " ."


def apply_nms(boxes: torch.Tensor, logits: torch.Tensor, phrases: list,
              iou_threshold: float = 0.5) -> tuple:
    """
    应用非极大值抑制去除重叠框
    """
    if len(boxes) == 0:
        return boxes, logits, phrases

    # boxes 是归一化的 [cx, cy, w, h] 格式，转换为 [x1, y1, x2, y2]
    boxes_xyxy = box_cxcywh_to_xyxy(boxes)

    keep_indices = ops.nms(boxes_xyxy, logits, iou_threshold)

    return (boxes[keep_indices],
            logits[keep_indices],
            [phrases[i] for i in keep_indices.tolist()])


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    将 [cx, cy, w, h] 格式转换为 [x1, y1, x2, y2] 格式
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def segment_with_text(
        image_path: str,
        text_prompt: str,
        box_threshold: float = 0.30,  # 提高默认阈值
        text_threshold: float = 0.25,  # 提高默认阈值
        min_score: float = 0.35,  # logits 过滤阈值
        min_area: int = 500,  # 掩码最小像素数过滤
        nms_threshold: float = 0.5,  # NMS IoU 阈值
):
    """
    使用 GroundingDINO + SAM 进行文本引导的图像分割

    Args:
        image_path: 图片路径
        text_prompt: 文本提示，多个类别用空格、逗号或句号分隔
        box_threshold: GroundingDINO 框置信度阈值
        text_threshold: GroundingDINO 文本匹配阈值
        min_score: 最低得分过滤
        min_area: 最小掩码面积
        nms_threshold: NMS IoU 阈值

    Returns:
        annotated, detections, masks_np, phrases, image_source
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 规范化提示词
    prompt = normalize_prompt(text_prompt)
    if not prompt:
        raise ValueError("text_prompt is empty")

    logger.info("Normalized prompt: '%s'", prompt)
    logger.info("Loading image %s", image_path)

    image_source, image = load_image(str(path))
    H, W, _ = image_source.shape
    logger.info("Image size: %d x %d", W, H)

    logger.info("Running GroundingDINO (box_thresh=%.2f, text_thresh=%.2f)",
                box_threshold, text_threshold)

    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=grounding_dino_model,
            image=image,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=DEVICE,  # 显式指定设备
        )

    if boxes is None or len(boxes) == 0:
        logger.warning("No boxes found for prompt: %s", prompt)
        return None, None, None, None, None

    logger.info("Found %d initial boxes", len(boxes))
    for i, (box, score, phrase) in enumerate(zip(boxes, logits, phrases)):
        logger.info("  Box %d: %s, score=%.3f, phrase='%s'", i, box.tolist(), score.item(), phrase)

    # 应用 NMS 去除重叠框
    boxes, logits, phrases = apply_nms(boxes, logits, phrases, nms_threshold)
    logger.info("After NMS: %d boxes", len(boxes))

    # logits 过滤：保留得分较高的框
    keep = logits >= min_score
    if keep.sum() == 0:
        logger.warning("No boxes after score filter (min_score=%.2f)", min_score)
        return None, None, None, None, None

    boxes = boxes[keep]
    logits = logits[keep]
    phrases = [phrases[i] for i, k in enumerate(keep.tolist()) if k]
    logger.info("After score filter: %d boxes", len(boxes))

    # 将归一化框转换为像素坐标 [x1, y1, x2, y2]
    boxes_xyxy = box_cxcywh_to_xyxy(boxes) * torch.tensor([W, H, W, H], device=boxes.device)

    # 确保框在图像范围内
    boxes_xyxy[:, 0] = boxes_xyxy[:, 0].clamp(0, W)
    boxes_xyxy[:, 1] = boxes_xyxy[:, 1].clamp(0, H)
    boxes_xyxy[:, 2] = boxes_xyxy[:, 2].clamp(0, W)
    boxes_xyxy[:, 3] = boxes_xyxy[:, 3].clamp(0, H)

    logger.info("Running SAM on %d boxes", len(boxes_xyxy))

    # SAM 预测
    sam_predictor.set_image(image_source)

    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        boxes_xyxy.to(DEVICE),
        image_source.shape[:2]
    )

    with torch.no_grad():
        masks, scores, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=True,  # 输出多个掩码选项
        )

    # masks shape: [N, 3, H, W], scores shape: [N, 3]
    # 选择每个框得分最高的掩码
    best_mask_indices = scores.argmax(dim=1)
    masks = torch.stack([masks[i, best_mask_indices[i]] for i in range(len(masks))])
    masks = masks.unsqueeze(1)  # [N, 1, H, W]

    # 掩码面积过滤
    masks_np = masks.cpu().numpy()  # [N,1,H,W]
    mask_areas = masks_np.sum(axis=(1, 2, 3))

    logger.info("Mask areas: %s", mask_areas.tolist())

    keep = mask_areas >= min_area
    if keep.sum() == 0:
        logger.warning("No masks after area filter (min_area=%d)", min_area)
        return None, None, None, None, None

    masks_np = masks_np[keep]
    boxes_xyxy_np = boxes_xyxy.cpu().numpy()[keep]
    logits_np = logits.cpu().numpy()[keep]
    phrases = [phrases[i] for i, k in enumerate(keep.tolist()) if k]

    logger.info("Final: %d detections", len(phrases))

    detections = sv.Detections(
        xyxy=boxes_xyxy_np,
        mask=masks_np.squeeze(1),
        class_id=np.zeros(len(boxes_xyxy_np), dtype=int),
        confidence=logits_np,
    )

    # 可视化
    box_annotator = sv.BoxAnnotator(thickness=2)
    mask_annotator = sv.MaskAnnotator(opacity=0.4)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    # 创建带置信度的标签
    labels = [f"{phrase} {conf:.2f}" for phrase, conf in zip(phrases, logits_np)]

    annotated = mask_annotator.annotate(scene=image_source.copy(), detections=detections)
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    return annotated, detections, masks_np, phrases, image_source


def save_outputs(image_source: np.ndarray, annotated: np.ndarray, detections: sv.Detections,
                 masks: np.ndarray, output_vis: Path, output_mask_dir: Path, phrases):
    """保存所有输出文件"""
    output_vis.parent.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    # 可视化
    cv2.imwrite(str(output_vis), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    logger.info("已保存可视化到 %s", output_vis)

    # 保存全部掩码为 npy
    np.save(output_mask_dir / "masks.npy", masks)
    logger.info("已保存所有掩码 numpy 到 %s", output_mask_dir / "masks.npy")

    H, W, _ = image_source.shape
    masks_np = masks[:, 0]  # [N, H, W]

    for idx, mask_bin in enumerate(masks_np):
        # 二值 PNG
        mask_path = output_mask_dir / f"mask_{idx:03d}.png"
        mask_uint8 = (mask_bin * 255).astype(np.uint8)
        cv2.imwrite(str(mask_path), mask_uint8)

        # 透明抠图（原图背景透明）
        rgba = cv2.cvtColor(image_source, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = mask_uint8
        # 只保留掩码区域的颜色
        for c in range(3):
            rgba[:, :, c] = (rgba[:, :, c] * mask_bin).astype(np.uint8)
        keep_path = output_mask_dir / f"cutout_{idx:03d}.png"
        cv2.imwrite(str(keep_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

        # 按框裁剪原图
        x1, y1, x2, y2 = detections.xyxy[idx].round().astype(int)
        x1 = np.clip(x1, 0, W)
        x2 = np.clip(x2, 0, W)
        y1 = np.clip(y1, 0, H)
        y2 = np.clip(y2, 0, H)

        if x2 <= x1 or y2 <= y1:
            logger.warning("skip empty crop for idx %d, box %s", idx, detections.xyxy[idx])
            continue

        crop = image_source[y1:y2, x1:x2]
        if crop.size == 0:
            logger.warning("skip empty crop (size=0) for idx %d", idx)
            continue

        crop_path = output_mask_dir / f"crop_{idx:03d}.png"
        cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        # 记录标签和置信度
        if phrases is not None:
            txt_path = output_mask_dir / f"info_{idx:03d}.txt"
            conf = detections.confidence[idx] if detections.confidence is not None else 0
            txt_path.write_text(
                f"label: {phrases[idx]}\n"
                f"confidence: {conf:.4f}\n"
                f"box: {detections.xyxy[idx].tolist()}\n"
                f"mask_area: {mask_bin.sum()}\n",
                encoding="utf-8"
            )

    logger.info("已逐个保存掩码/抠图/裁剪到 %s", output_mask_dir)


def main():
    image_path = Path(
        "/home/cz/software/pycharm-2024.2.3/PycharmProjects/python-utils/assets/grounding_dino_sam/image/input/002.jpeg")
    text_prompt = "person"  # 使用逗号分隔更清晰
    output_vis = Path(
        "/home/cz/software/pycharm-2024.2.3/PycharmProjects/python-utils/assets/grounding_dino_sam/image/output/grounded_sam_result_002.jpg")
    output_mask_dir = Path(
        "/home/cz/software/pycharm-2024.2.3/PycharmProjects/python-utils/assets/grounding_dino_sam/image/output/masks")

    if not image_path.is_file():
        raise FileNotFoundError(f"找不到图片: {image_path}")

    annotated, detections, masks, phrases, image_source = segment_with_text(
        image_path=str(image_path),
        text_prompt=text_prompt,
        box_threshold=0.30,  # 适当提高阈值过滤低质量检测
        text_threshold=0.25,
        min_score=0.35,
        min_area=500,
        nms_threshold=0.5,  # NMS 阈值
    )

    if annotated is None:
        logger.warning("未检测到目标，未生成输出")
        return

    save_outputs(image_source, annotated, detections, masks, output_vis, output_mask_dir, phrases)
    logger.info("处理完成!")


if __name__ == "__main__":
    main()