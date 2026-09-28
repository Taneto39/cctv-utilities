#!/usr/bin/env python3
"""
Quick benchmark: how fast can decord decode this camera's video on the GPU
(NVDEC), compared to plain OpenCV CPU decode?

This does NOT run motion detection -- it only measures raw decode speed, so
we can tell whether GPU decode is actually worth building into motion_scan.py
before investing more time in it.

Usage:
    pip install decord --break-system-packages   (or just: pip install decord)
    python bench_decord_gpu.py <file path>
"""
import sys
import time

import cv2

path = sys.argv[1]
N = 500  # how many frames to time


def bench_cpu():
    cap = cv2.VideoCapture(path)
    t0 = time.time()
    n = 0
    while n < N:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
    dt = time.time() - t0
    cap.release()
    print(f"[OpenCV CPU]    {n} frames in {dt:.2f}s -> {n / dt:.1f} fps")


def bench_decord_gpu():
    try:
        import decord
    except ImportError:
        print("[decord]        not installed -- run: pip install decord")
        return
    try:
        ctx = decord.gpu(0)
        vr = decord.VideoReader(path, ctx=ctx)
    except Exception as e:
        print(f"[decord GPU]    failed to open with GPU context: {e}")
        return
    t0 = time.time()
    n = min(N, len(vr))
    for i in range(n):
        _ = vr[i]
    dt = time.time() - t0
    print(f"[decord GPU]    {n} frames in {dt:.2f}s -> {n / dt:.1f} fps")


def bench_decord_cpu():
    try:
        import decord
    except ImportError:
        return
    try:
        vr = decord.VideoReader(path, ctx=decord.cpu(0))
    except Exception as e:
        print(f"[decord CPU]    failed: {e}")
        return
    t0 = time.time()
    n = min(N, len(vr))
    for i in range(n):
        _ = vr[i]
    dt = time.time() - t0
    print(f"[decord CPU]    {n} frames in {dt:.2f}s -> {n / dt:.1f} fps")


if __name__ == "__main__":
    print(f"Benchmarking {path} ({N} frames each)...")
    bench_cpu()
    bench_decord_cpu()
    bench_decord_gpu()
