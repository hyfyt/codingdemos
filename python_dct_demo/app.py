#!/usr/bin/env python3
"""Interactive JPEG DCT transform-coding demo.

This is a small dependency-light Python web demo.  It uses the Python
standard-library HTTP server for the UI and NumPy/Pillow for image math.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
BLOCK = 8
MAX_DEMO_DIMENSION = 768


LUMINANCE_QUANTISATION_TABLE = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float64,
)


CHROMA_QUANTISATION_TABLE = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=np.float64,
)


def dct_matrix(size: int = BLOCK) -> np.ndarray:
    """Return an orthonormal DCT-II transform matrix."""
    matrix = np.empty((size, size), dtype=np.float64)
    factor = math.pi / (2.0 * size)
    for k in range(size):
        alpha = math.sqrt(1.0 / size) if k == 0 else math.sqrt(2.0 / size)
        for n in range(size):
            matrix[k, n] = alpha * math.cos((2 * n + 1) * k * factor)
    return matrix


DCT_MATRIX = dct_matrix()


def round_half_away_from_zero(values: np.ndarray) -> np.ndarray:
    """Round positive and negative halves away from zero."""
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def scaled_quantisation_table(table: np.ndarray, quality_factor: int) -> np.ndarray:
    """Port of TransformCoding.qualityFactorToQuantisationTable."""
    quality = int(quality_factor)
    if quality < 1:
        quality = 1
    elif quality > 100:
        quality = 100

    if quality < 50:
        scale_factor = 5000 / quality
    else:
        scale_factor = 200 - quality * 2

    scaled = round_half_away_from_zero(((table * scale_factor) + 50) / 100)
    scaled[scaled <= 0] = 1
    scaled[scaled > 255] = 255
    return scaled.astype(np.float64)


def pad_to_block(channel: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    height, width = channel.shape
    padded_height = int(math.ceil(height / BLOCK) * BLOCK)
    padded_width = int(math.ceil(width / BLOCK) * BLOCK)
    pad_bottom = padded_height - height
    pad_right = padded_width - width
    if pad_bottom == 0 and pad_right == 0:
        return channel.astype(np.float64), (height, width)
    padded = np.pad(channel, ((0, pad_bottom), (0, pad_right)), mode="edge")
    return padded.astype(np.float64), (height, width)


def blocks_from_channel(channel: np.ndarray) -> np.ndarray:
    height, width = channel.shape
    return channel.reshape(height // BLOCK, BLOCK, width // BLOCK, BLOCK).transpose(0, 2, 1, 3)


def channel_from_blocks(blocks: np.ndarray) -> np.ndarray:
    block_rows, block_cols = blocks.shape[:2]
    return blocks.transpose(0, 2, 1, 3).reshape(block_rows * BLOCK, block_cols * BLOCK)


def dct2_blocks(blocks: np.ndarray) -> np.ndarray:
    return np.einsum("ux,...xy,vy->...uv", DCT_MATRIX, blocks, DCT_MATRIX, optimize=True)


def idct2_blocks(blocks: np.ndarray) -> np.ndarray:
    return np.einsum("ux,...uv,vy->...xy", DCT_MATRIX, blocks, DCT_MATRIX, optimize=True)


def encode_channel(
    channel: np.ndarray, quantisation_table: np.ndarray, coefficient_mask: np.ndarray
) -> Dict[str, np.ndarray]:
    padded, original_shape = pad_to_block(channel)
    shifted = padded - 128.0
    source_blocks = blocks_from_channel(shifted)

    dct_blocks = dct2_blocks(source_blocks)
    quantised_blocks = round_half_away_from_zero(dct_blocks / quantisation_table)
    masked_quantised_blocks = quantised_blocks * coefficient_mask
    dequantised_blocks = masked_quantised_blocks * quantisation_table
    inverse_blocks = idct2_blocks(dequantised_blocks)

    coefficients = channel_from_blocks(dct_blocks)
    quantised = channel_from_blocks(masked_quantised_blocks)
    dequantised = channel_from_blocks(dequantised_blocks)
    inverse_shifted = channel_from_blocks(inverse_blocks)
    reconstructed = np.clip(round_half_away_from_zero(inverse_shifted + 128.0), 0, 255)

    height, width = original_shape
    return {
        "padded_input": padded,
        "coefficients": coefficients,
        "quantised": quantised,
        "dequantised": dequantised,
        "inverse_shifted": inverse_shifted,
        "reconstructed": reconstructed,
        "cropped_reconstructed": reconstructed[:height, :width],
    }


def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    data = rgb.astype(np.float64)
    r = data[:, :, 0]
    g = data[:, :, 1]
    b = data[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack([y, cb, cr], axis=2)


def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    y = ycbcr[:, :, 0]
    cb = ycbcr[:, :, 1] - 128.0
    cr = ycbcr[:, :, 2] - 128.0
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    rgb = np.stack([r, g, b], axis=2)
    return np.clip(round_half_away_from_zero(rgb), 0, 255).astype(np.uint8)


def compute_psnr(reference: np.ndarray, reconstructed: np.ndarray) -> str:
    diff = reference.astype(np.float64) - reconstructed.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse == 0:
        return "inf"
    return f"{20 * math.log10(255.0 / math.sqrt(mse)):.2f}"


def matrix_for_table(values: np.ndarray) -> List[List[int]]:
    rounded = round_half_away_from_zero(values).astype(np.int64)
    return rounded.tolist()


def image_to_png_base64(image_array: np.ndarray) -> str:
    image = Image.fromarray(image_array.astype(np.uint8), "RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def grayscale_to_png_base64(values: np.ndarray, size: int = 52) -> str:
    normalized = values.astype(np.float64)
    normalized = normalized - float(np.min(normalized))
    peak = float(np.max(normalized))
    if peak == 0:
        normalized = np.ones_like(normalized) * 255
    else:
        normalized = normalized / peak * 255
    image = Image.fromarray(normalized.astype(np.uint8), "L")
    image = image.resize((size, size), Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST)
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def basis_images() -> List[str]:
    bases = []
    for row in range(BLOCK):
        for col in range(BLOCK):
            coefficients = np.zeros((BLOCK, BLOCK), dtype=np.float64)
            coefficients[row, col] = 1.0
            basis = DCT_MATRIX.T @ coefficients @ DCT_MATRIX
            bases.append(grayscale_to_png_base64(basis))
    return bases


def list_example_images() -> List[str]:
    allowed = {".bmp", ".jpg", ".jpeg", ".png"}
    files = [item.name for item in EXAMPLES_DIR.iterdir() if item.suffix.lower() in allowed and item.is_file()]
    return sorted(files, key=lambda name: (Path(name).suffix.lower() != ".bmp", name.lower()))


def safe_example_path(name: str) -> Path:
    clean_name = Path(name).name
    if clean_name != name:
        raise ValueError("Invalid example image name.")
    path = EXAMPLES_DIR / clean_name
    if not path.exists() or path.suffix.lower() not in {".bmp", ".jpg", ".jpeg", ".png"}:
        raise ValueError("Example image not found.")
    return path


def load_demo_image(payload: Dict[str, Any]) -> np.ndarray:
    if payload.get("image_data"):
        data_url = str(payload["image_data"])
        if "," in data_url:
            _, encoded = data_url.split(",", 1)
        else:
            encoded = data_url
        image_bytes = base64.b64decode(encoded)
        image = Image.open(BytesIO(image_bytes))
    else:
        image_name = str(payload.get("image_name") or list_example_images()[0])
        image = Image.open(safe_example_path(image_name))

    image = image.convert("RGB")
    width, height = image.size
    longest = max(width, height)
    if longest > MAX_DEMO_DIMENSION:
        scale = MAX_DEMO_DIMENSION / longest
        resampling = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        image = image.resize((int(width * scale), int(height * scale)), resampling)
    return np.asarray(image, dtype=np.uint8)


def normalise_mask(raw_mask: Any) -> np.ndarray:
    if not isinstance(raw_mask, list):
        return np.ones((BLOCK, BLOCK), dtype=np.float64)
    mask = np.array(raw_mask, dtype=np.float64)
    if mask.shape != (BLOCK, BLOCK):
        return np.ones((BLOCK, BLOCK), dtype=np.float64)
    return (mask > 0).astype(np.float64)


def clamp_block_start(value: Any, limit: int) -> int:
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        numeric = 0
    numeric = max(0, numeric)
    snapped = (numeric // BLOCK) * BLOCK
    max_start = max(0, limit - BLOCK)
    snapped_max = (max_start // BLOCK) * BLOCK
    return min(snapped, snapped_max)


def process_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    rgb = load_demo_image(payload)
    height, width = rgb.shape[:2]
    quality = int(payload.get("quality", 60))
    coefficient_mask = normalise_mask(payload.get("mask"))

    ycbcr = rgb_to_ycbcr(rgb)
    luminance_q = scaled_quantisation_table(LUMINANCE_QUANTISATION_TABLE, quality)
    chroma_q = scaled_quantisation_table(CHROMA_QUANTISATION_TABLE, quality)
    tables = [luminance_q, chroma_q, chroma_q]

    encoded_channels = [
        encode_channel(ycbcr[:, :, channel_index], tables[channel_index], coefficient_mask)
        for channel_index in range(3)
    ]
    reconstructed_ycbcr = np.stack(
        [channel["cropped_reconstructed"] for channel in encoded_channels],
        axis=2,
    )
    reconstructed_rgb = ycbcr_to_rgb(reconstructed_ycbcr)

    block_x = clamp_block_start(payload.get("block_x", 0), width)
    block_y = clamp_block_start(payload.get("block_y", 0), height)
    y_channel = encoded_channels[0]
    block_slice = np.s_[block_y : block_y + BLOCK, block_x : block_x + BLOCK]

    block_tables = {
        "input_y": matrix_for_table(y_channel["padded_input"][block_slice]),
        "dequantised": matrix_for_table(y_channel["dequantised"][block_slice]),
        "dct": matrix_for_table(y_channel["coefficients"][block_slice]),
        "inverse": matrix_for_table(y_channel["inverse_shifted"][block_slice]),
        "quantised": matrix_for_table(y_channel["quantised"][block_slice]),
        "output_y": matrix_for_table(y_channel["reconstructed"][block_slice]),
    }

    return {
        "width": width,
        "height": height,
        "input_image": image_to_png_base64(rgb),
        "output_image": image_to_png_base64(reconstructed_rgb),
        "psnr": compute_psnr(ycbcr[:, :, 0], reconstructed_ycbcr[:, :, 0]),
        "block_x": block_x,
        "block_y": block_y,
        "selected_count": int(np.sum(coefficient_mask)),
        "quality": quality,
        "tables": block_tables,
        "quantisation_table": matrix_for_table(luminance_q),
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Transform Coding: DCT</title>
  <style>
    :root {
      --ink: #20242a;
      --muted: #68707a;
      --line: #cfd5dc;
      --panel: #ffffff;
      --panel-alt: #f4f6f8;
      --accent: #2563a8;
      --accent-soft: #dcecff;
      --green: #14785a;
      --warning: #b45309;
      --shadow: 0 8px 24px rgba(20, 27, 37, 0.08);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background: #eef1f5;
      font-family: Arial, Helvetica, sans-serif;
      letter-spacing: 0;
    }

    button, input, select {
      font: inherit;
    }

    .topbar {
      height: 42px;
      display: flex;
      align-items: center;
      padding: 0 18px;
      color: #111827;
      background: #d9dde3;
      border-bottom: 1px solid #b9c0ca;
      font-size: 18px;
      font-weight: 700;
    }

    .app {
      padding: 14px;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 14px;
    }

    .workspace {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) minmax(330px, 0.95fr) minmax(260px, 1fr);
      gap: 14px;
      align-items: start;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      border-radius: 6px;
      overflow: hidden;
    }

    .panel-header {
      min-height: 40px;
      padding: 10px 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      background: var(--panel-alt);
      border-bottom: 1px solid var(--line);
      font-weight: 700;
    }

    .panel-header .sub {
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
      white-space: nowrap;
    }

    .image-body {
      min-height: 320px;
      padding: 12px;
      display: grid;
      place-items: center;
      background: #ffffff;
    }

    canvas.image-canvas {
      display: block;
      max-width: 100%;
      max-height: 58vh;
      width: auto;
      height: auto;
      border: 1px solid #aeb6c1;
      background: #f7f8fa;
      image-rendering: auto;
      cursor: crosshair;
    }

    #outputCanvas {
      cursor: default;
    }

    .control-body {
      padding: 12px;
      display: grid;
      gap: 12px;
    }

    .field {
      display: grid;
      gap: 6px;
    }

    .field-row {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }

    label {
      color: #26313d;
      font-size: 13px;
      font-weight: 700;
    }

    select, input[type="file"] {
      min-height: 34px;
      width: 100%;
      border: 1px solid #adb6c2;
      border-radius: 4px;
      background: #fff;
      padding: 5px 7px;
      color: var(--ink);
    }

    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }

    .quality-line {
      display: grid;
      grid-template-columns: 36px 1fr 36px;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
    }

    .basis-title {
      margin-top: 2px;
      font-size: 13px;
      font-weight: 700;
    }

    .basis-grid {
      display: grid;
      grid-template-columns: repeat(8, minmax(0, 1fr));
      gap: 4px;
      background: #e7ebf0;
      border: 1px solid #b9c1cb;
      padding: 6px;
      border-radius: 4px;
    }

    .basis-button {
      aspect-ratio: 1 / 1;
      min-width: 0;
      padding: 0;
      border: 2px solid #2d333b;
      border-radius: 3px;
      background: #ffffff center / cover no-repeat;
      cursor: pointer;
      transition: transform 90ms ease, border-color 90ms ease, opacity 90ms ease;
    }

    .basis-button:hover {
      transform: translateY(-1px);
      border-color: var(--accent);
    }

    .basis-button.is-off {
      background-image: none !important;
      background-color: #f9fafb;
      opacity: 0.45;
      border-color: #a5adba;
    }

    .button-row {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }

    .command {
      min-height: 34px;
      border: 1px solid #9aa6b4;
      border-radius: 4px;
      background: #fff;
      color: #1d2733;
      cursor: pointer;
      font-weight: 700;
    }

    .command:hover {
      border-color: var(--accent);
      color: var(--accent);
    }

    .status-line {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      min-height: 34px;
      padding: 7px 9px;
      background: #f7f9fb;
      border: 1px solid #c8d0da;
      border-radius: 4px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }

    .status-line strong {
      color: var(--green);
    }

    .tables-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .tables-body {
      padding: 12px;
      display: grid;
      grid-template-columns: repeat(3, minmax(230px, 1fr));
      gap: 12px;
    }

    .matrix-wrap {
      min-width: 0;
    }

    .matrix-title {
      margin: 0 0 6px;
      color: #26313d;
      font-size: 13px;
      font-weight: 700;
    }

    table.matrix {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-family: "Courier New", Courier, monospace;
      font-size: 12px;
      background: #fff;
      border: 1px solid #b9c1cb;
    }

    table.matrix td {
      width: 12.5%;
      padding: 4px 3px;
      text-align: right;
      border: 1px solid #d7dde5;
      overflow: hidden;
      text-overflow: clip;
      white-space: nowrap;
    }

    table.matrix tr:nth-child(even) td {
      background: #f5f7fa;
    }

    .busy {
      opacity: 0.65;
      pointer-events: none;
    }

    .error {
      color: #a03232;
      font-weight: 700;
    }

    @media (max-width: 1080px) {
      .workspace {
        grid-template-columns: 1fr;
      }

      .tables-body {
        grid-template-columns: repeat(2, minmax(220px, 1fr));
      }

      canvas.image-canvas {
        max-height: none;
      }
    }

    @media (max-width: 680px) {
      .app {
        padding: 10px;
      }

      .tables-body {
        grid-template-columns: 1fr;
      }

      .button-row {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <header class="topbar">Transform Coding: DCT</header>

  <main class="app" id="appRoot">
    <section class="workspace">
      <section class="panel">
        <div class="panel-header">
          <span>Input Image</span>
          <span class="sub" id="blockLabel">Block: 0, 0</span>
        </div>
        <div class="image-body">
          <canvas id="inputCanvas" class="image-canvas"></canvas>
        </div>
      </section>

      <section class="panel">
        <div class="panel-header">
          <span>Click to select DCT bases</span>
          <span class="sub" id="coefficientCount">64 / 64</span>
        </div>
        <div class="control-body">
          <div class="field">
            <label for="imageSelect">Input Image</label>
            <select id="imageSelect"></select>
          </div>

          <div class="field">
            <label for="uploadInput">Image Read</label>
            <input id="uploadInput" type="file" accept="image/*">
          </div>

          <div class="field">
            <div class="field-row">
              <label for="qualitySlider">JPEG Quality Factor</label>
              <strong id="qualityValue">60</strong>
            </div>
            <input id="qualitySlider" type="range" min="0" max="100" value="60">
            <div class="quality-line"><span>0</span><span></span><span>100</span></div>
          </div>

          <div class="basis-grid" id="basisGrid"></div>

          <div class="button-row">
            <button class="command" id="setAllButton" type="button">Set All</button>
            <button class="command" id="clearAllButton" type="button">Remove All</button>
            <button class="command" id="lowPassButton" type="button">Low Pass</button>
          </div>

          <div class="status-line">
            <span id="statusText">Ready</span>
            <span>PSNR: <strong id="psnrValue">--</strong> dB</span>
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-header">
          <span>Output Image</span>
          <span class="sub" id="qualityLabel">Quality: 60</span>
        </div>
        <div class="image-body">
          <canvas id="outputCanvas" class="image-canvas"></canvas>
        </div>
      </section>
    </section>

    <section class="tables-panel">
      <div class="panel-header">
        <span>Selected Block</span>
        <span class="sub" id="tableBlockLabel">8 x 8</span>
      </div>
      <div class="tables-body" id="tablesBody"></div>
    </section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    const state = {
      images: [],
      imageName: "",
      imageData: null,
      quality: 60,
      blockX: 0,
      blockY: 0,
      mask: Array.from({ length: 8 }, () => Array(8).fill(1)),
    };
    let lastResponse = null;
    let processTimer = null;

    const tableSpecs = [
      ["Input Y Pixel Values", "input_y"],
      ["Dequantised DCT Coefficients", "dequantised"],
      ["DCT Coefficients", "dct"],
      ["Inverse Transform Coefficients", "inverse"],
      ["Quantised DCT Coefficients", "quantised"],
      ["Output Y Pixel Values", "output_y"],
    ];

    function debounceProcess() {
      window.clearTimeout(processTimer);
      processTimer = window.setTimeout(processImage, 140);
    }

    function setBusy(isBusy) {
      $("appRoot").classList.toggle("busy", isBusy);
      if (isBusy) {
        $("statusText").textContent = "Updating";
      }
    }

    function setStatus(text, isError = false) {
      const node = $("statusText");
      node.textContent = text;
      node.classList.toggle("error", isError);
    }

    async function init() {
      wireControls();
      await loadImages();
      await loadBases();
      await processImage();
    }

    function wireControls() {
      $("imageSelect").addEventListener("change", (event) => {
        state.imageName = event.target.value;
        state.imageData = null;
        $("uploadInput").value = "";
        state.blockX = 0;
        state.blockY = 0;
        processImage();
      });

      $("uploadInput").addEventListener("change", (event) => {
        const file = event.target.files && event.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = () => {
          state.imageData = reader.result;
          state.blockX = 0;
          state.blockY = 0;
          processImage();
        };
        reader.readAsDataURL(file);
      });

      $("qualitySlider").addEventListener("input", (event) => {
        state.quality = Number(event.target.value);
        $("qualityValue").textContent = String(state.quality);
        $("qualityLabel").textContent = `Quality: ${state.quality}`;
        debounceProcess();
      });

      $("setAllButton").addEventListener("click", () => {
        state.mask = Array.from({ length: 8 }, () => Array(8).fill(1));
        syncBasisButtons();
        processImage();
      });

      $("clearAllButton").addEventListener("click", () => {
        state.mask = Array.from({ length: 8 }, () => Array(8).fill(0));
        syncBasisButtons();
        processImage();
      });

      $("lowPassButton").addEventListener("click", () => {
        state.mask = Array.from({ length: 8 }, (_, row) =>
          Array.from({ length: 8 }, (_, col) => (row + col <= 4 ? 1 : 0))
        );
        syncBasisButtons();
        processImage();
      });

      $("inputCanvas").addEventListener("click", (event) => {
        if (!lastResponse) return;
        const rect = event.currentTarget.getBoundingClientRect();
        const imageX = Math.floor((event.clientX - rect.left) * lastResponse.width / rect.width);
        const imageY = Math.floor((event.clientY - rect.top) * lastResponse.height / rect.height);
        state.blockX = Math.floor(Math.max(0, imageX) / 8) * 8;
        state.blockY = Math.floor(Math.max(0, imageY) / 8) * 8;
        processImage();
      });
    }

    async function loadImages() {
      const response = await fetch("/api/images");
      const payload = await response.json();
      state.images = payload.images || [];
      const select = $("imageSelect");
      select.innerHTML = "";
      state.images.forEach((name) => {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        select.appendChild(option);
      });
      state.imageName = state.images[0] || "";
      select.value = state.imageName;
    }

    async function loadBases() {
      const response = await fetch("/api/bases");
      const payload = await response.json();
      const grid = $("basisGrid");
      grid.innerHTML = "";
      payload.bases.forEach((image, index) => {
        const row = Math.floor(index / 8);
        const col = index % 8;
        const button = document.createElement("button");
        button.type = "button";
        button.className = "basis-button";
        button.title = `(${row}, ${col})`;
        button.style.backgroundImage = `url(data:image/png;base64,${image})`;
        button.dataset.row = String(row);
        button.dataset.col = String(col);
        button.addEventListener("click", () => {
          state.mask[row][col] = state.mask[row][col] ? 0 : 1;
          syncBasisButtons();
          processImage();
        });
        grid.appendChild(button);
      });
      syncBasisButtons();
    }

    function syncBasisButtons() {
      document.querySelectorAll(".basis-button").forEach((button) => {
        const row = Number(button.dataset.row);
        const col = Number(button.dataset.col);
        button.classList.toggle("is-off", !state.mask[row][col]);
      });
      const count = state.mask.flat().reduce((sum, value) => sum + Number(value), 0);
      $("coefficientCount").textContent = `${count} / 64`;
    }

    async function processImage() {
      setBusy(true);
      try {
        const response = await fetch("/api/process", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image_name: state.imageName,
            image_data: state.imageData,
            quality: state.quality,
            mask: state.mask,
            block_x: state.blockX,
            block_y: state.blockY,
          }),
        });
        if (!response.ok) {
          const text = await response.text();
          throw new Error(text || response.statusText);
        }
        lastResponse = await response.json();
        state.blockX = lastResponse.block_x;
        state.blockY = lastResponse.block_y;
        renderResponse(lastResponse);
        setStatus("Ready");
      } catch (error) {
        setStatus(error.message || "Error", true);
      } finally {
        setBusy(false);
      }
    }

    function renderResponse(payload) {
      $("psnrValue").textContent = payload.psnr;
      $("qualityLabel").textContent = `Quality: ${payload.quality}`;
      $("blockLabel").textContent = `Block: ${payload.block_x}, ${payload.block_y}`;
      $("tableBlockLabel").textContent = `(${payload.block_x}, ${payload.block_y})`;
      $("coefficientCount").textContent = `${payload.selected_count} / 64`;
      drawImageCanvas($("inputCanvas"), payload.input_image, payload, true);
      drawImageCanvas($("outputCanvas"), payload.output_image, payload, true);
      renderTables(payload.tables);
    }

    function drawImageCanvas(canvas, base64Image, payload, drawSelection) {
      const image = new Image();
      image.onload = () => {
        canvas.width = payload.width;
        canvas.height = payload.height;
        const context = canvas.getContext("2d");
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.drawImage(image, 0, 0, payload.width, payload.height);
        if (drawSelection) {
          context.save();
          context.strokeStyle = "#111827";
          context.lineWidth = Math.max(2, Math.round(Math.min(payload.width, payload.height) / 180));
          context.strokeRect(payload.block_x + 0.5, payload.block_y + 0.5, 8, 8);
          context.restore();
        }
      };
      image.src = `data:image/png;base64,${base64Image}`;
    }

    function renderTables(tables) {
      const body = $("tablesBody");
      body.innerHTML = "";
      tableSpecs.forEach(([title, key]) => {
        const wrap = document.createElement("div");
        wrap.className = "matrix-wrap";
        const heading = document.createElement("h3");
        heading.className = "matrix-title";
        heading.textContent = title;
        const table = document.createElement("table");
        table.className = "matrix";
        const tbody = document.createElement("tbody");
        (tables[key] || []).forEach((row) => {
          const tr = document.createElement("tr");
          row.forEach((value) => {
            const td = document.createElement("td");
            td.textContent = String(value);
            tr.appendChild(td);
          });
          tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        wrap.appendChild(heading);
        wrap.appendChild(table);
        body.appendChild(wrap);
      });
    }

    init();
  </script>
</body>
</html>
"""


class DCTDemoHandler(BaseHTTPRequestHandler):
    server_version = "DCTDemo/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_bytes(self, content: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        self.send_bytes(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8", status)

    def send_error_text(self, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
        self.send_bytes(message.encode("utf-8"), "text/plain; charset=utf-8", status)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/images":
            self.send_json({"images": list_example_images()})
        elif path == "/api/bases":
            self.send_json({"bases": basis_images()})
        elif path == "/favicon.ico":
            self.send_bytes(b"", "image/x-icon", HTTPStatus.NO_CONTENT)
        elif path.startswith("/examples/"):
            self.serve_example(path)
        else:
            self.send_error_text("Not found", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/process":
            self.send_error_text("Not found", HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length)
            payload = json.loads(raw_body.decode("utf-8"))
            result = process_payload(payload)
        except Exception as exc:  # noqa: BLE001 - show concise demo errors in UI
            self.send_error_text(str(exc), HTTPStatus.BAD_REQUEST)
            return

        self.send_json(result)

    def serve_example(self, request_path: str) -> None:
        try:
            name = unquote(request_path.removeprefix("/examples/"))
            path = safe_example_path(name)
        except ValueError as exc:
            self.send_error_text(str(exc), HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_bytes(path.read_bytes(), content_type)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Python DCT transform-coding demo.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", default=8501, type=int, help="Port to bind.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    httpd = ThreadingHTTPServer((args.host, args.port), DCTDemoHandler)
    print(f"DCT demo running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
