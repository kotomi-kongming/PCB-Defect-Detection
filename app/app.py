"""
Streamlit 可视化应用：上传 PCB 图片 -> 输出标注结果。

运行方式:
    streamlit run app/app.py

环境变量:
    PCB_WEIGHTS: (可选) 自定义模型权重路径，默认 models/best.pt
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO  # type: ignore

DEFAULT_WEIGHTS = Path(os.getenv("PCB_WEIGHTS", "models/best.pt"))


@st.cache_resource
def load_model(weights_path: str | Path) -> YOLO:
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"未找到模型权重: {weights_path}")
    return YOLO(str(weights_path))


def run_detection(model: YOLO, image: Image.Image, conf: float, iou: float, imgsz: int):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        results = model.predict(
            source=tmp.name,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            save=False,
            show=False,
        )
    result = results[0]
    annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    return result, annotated


def format_boxes(result) -> List[dict]:
    boxes = []
    if result.boxes is None:
        return boxes
    names = result.names
    for box in result.boxes:
        cls_id = int(box.cls[0])
        boxes.append(
            {
                "class": names.get(cls_id, str(cls_id)),
                "confidence": float(box.conf[0]),
                "x1": float(box.xyxy[0][0]),
                "y1": float(box.xyxy[0][1]),
                "x2": float(box.xyxy[0][2]),
                "y2": float(box.xyxy[0][3]),
            }
        )
    return boxes


def main() -> None:
    st.set_page_config(page_title="PCB 缺陷检测 Demo", layout="wide")
    st.title("PCB 缺陷检测可视化")
    st.caption("上传一张 PCB 图片，模型会输出标注结果。")

    weights_path = st.sidebar.text_input("模型权重路径", str(DEFAULT_WEIGHTS))
    conf = st.sidebar.slider("Confidence 阈值", 0.05, 0.95, 0.25, 0.05)
    iou = st.sidebar.slider("IoU 阈值", 0.1, 0.9, 0.5, 0.05)
    imgsz = st.sidebar.selectbox("推理尺寸 (imgsz)", [256, 320, 384, 512], index=0)

    @st.cache_resource(show_spinner=False)
    def _load(weights: str):
        return load_model(weights)

    model = None
    if weights_path:
        try:
            model = _load(weights_path)
        except Exception as exc:
            st.error(f"加载模型失败: {exc}")

    uploaded = st.file_uploader("上传 PCB 图片 (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded and model:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="原图", use_column_width=True)

        with st.spinner("推理中..."):
            result, annotated = run_detection(model, image, conf, iou, imgsz)

        st.subheader("检测结果")
        st.image(annotated, caption="标注图", use_column_width=True)

        boxes = format_boxes(result)
        if boxes:
            st.dataframe(boxes)
        else:
            st.info("未检测到缺陷，可调整阈值后重试。")
    elif not uploaded:
        st.info("请先上传一张 PCB 图片。")


if __name__ == "__main__":
    main()

