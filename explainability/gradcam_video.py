import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import tempfile
import os

from video_model.video_dataset import EchoVideoProcessor


def _find_last_conv(module):
    # Return the last nn.Conv3d module in a model
    import torch.nn as nn
    for m in reversed(list(module.modules())):
        if isinstance(m, nn.Conv3d):
            return m
    return None


def _gradcam_overlay_frames(video_path, model, target_class=1):
    device = next(model.parameters()).device
    processor = EchoVideoProcessor()
    clip = processor.preprocess(video_path).to(device)  # (1,C,T,H,W)

    target_module = _find_last_conv(model)
    if target_module is None:
        raise RuntimeError("No Conv3d layer found for Grad-CAM")

    activations = None
    gradients = None

    def forward_hook(module, inp, out):
        nonlocal activations
        activations = out.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()

    fh = target_module.register_forward_hook(forward_hook)
    bh = target_module.register_backward_hook(backward_hook)

    model.zero_grad()
    logits = model(clip)
    score = logits[0, target_class]
    score.backward(retain_graph=True)

    fh.remove()
    bh.remove()

    if activations is None or gradients is None:
        raise RuntimeError("Failed to capture activations or gradients")

    weights = gradients.mean(dim=(3, 4), keepdim=True)  # (1,C,T,1,1)
    cams = F.relu((weights * activations).sum(dim=1))  # (1,T,H,W)
    cams = cams[0].cpu().numpy()

    frames = processor.load_video(video_path)
    frames = processor.sample_clip(frames)

    overlay_frames = []
    for t in range(cams.shape[0]):
        heatmap = cams[t]
        heatmap = heatmap - heatmap.min()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        frame = frames[t]
        h, w, _ = frame.shape
        heat_resized = cv2.resize(heatmap, (w, h))
        heat_uint8 = np.uint8(255 * heat_resized)
        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

        overlay = 0.5 * frame.astype(np.float32) / 255.0 + 0.5 * (heat_color.astype(np.float32) / 255.0)
        overlay = np.clip(overlay, 0, 1)
        overlay_frames.append(np.uint8(overlay * 255))

    return overlay_frames


def gradcam_for_video(video_path, model, target_class=1):
    """Generate a Grad-CAM overlay image (PIL) for the center frame of the video clip."""
    overlay_frames = _gradcam_overlay_frames(video_path, model, target_class)
    center_frame = overlay_frames[len(overlay_frames) // 2]
    return Image.fromarray(center_frame)


def _write_overlay_video(overlay_frames, out_path, fps):
    h, w, _ = overlay_frames[0].shape
    suffix = os.path.splitext(out_path)[1].lower() if out_path else ".mp4"
    candidates = []
    if suffix == ".mp4":
        candidates.append(("mp4v", ".mp4"))
        candidates.append(("avc1", ".mp4"))
        candidates.append(("H264", ".mp4"))
        candidates.append(("MJPG", ".avi"))
    elif suffix == ".avi":
        candidates.append(("MJPG", ".avi"))
        candidates.append(("DIVX", ".avi"))
    else:
        candidates.append(("mp4v", ".mp4"))

    last_error = None
    for codec, ext in candidates:
        path = out_path or tempfile.NamedTemporaryFile(suffix=ext, delete=False).name
        if out_path is None:
            out_path = path
        else:
            path = out_path
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if not writer.isOpened():
            writer.release()
            last_error = f"codec {codec} could not be opened"
            continue
        for fr in overlay_frames:
            writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        writer.release()
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
        last_error = f"codec {codec} wrote an empty file"
    raise RuntimeError(f"Unable to write Grad-CAM video: {last_error}")


def gradcam_video(video_path, model, target_class=1, out_path=None, fps=10):
    """Generate a Grad-CAM overlay video for the sampled clip and return the file path.

    If `out_path` is provided, the result will be saved there; otherwise a temp file is used.
    """
    overlay_path = gradcam_for_video_internal(video_path, model, target_class, out_path, fps)
    return overlay_path


def gradcam_for_video_internal(video_path, model, target_class=1, out_path=None, fps=10):
    """Internal function that performs per-frame Grad-CAM and writes a video."""
    overlay_frames = _gradcam_overlay_frames(video_path, model, target_class)

    if out_path is None:
        out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    return _write_overlay_video(overlay_frames, out_path, fps)
