"""
Zero-shot turf detection using Grounding DINO (text-prompted box detection)
+ Segment Anything (box-prompted mask refinement), run via Hugging Face.

No training data involved at all — this exists specifically as a
generalization safeguard and comparison point against the fine-tuned U-Net:
agreement between the two on the external/unseen image is a much stronger
signal of correctness than either model's confidence alone, since they have
completely different failure modes (the U-Net can overfit this dataset's
domain; SAM/Grounding-DINO can be imprecise or miss subtle turf).

Requires: transformers, torch, Pillow.
    pip install transformers torch pillow
"""
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import (
    AutoProcessor, AutoModelForZeroShotObjectDetection,
    SamModel, SamProcessor,
)

from geoutils import load_image_and_transform

GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM_MODEL = "facebook/sam-vit-base"
TEXT_PROMPT = "grass. lawn. turf. mowed field."


def run_zero_shot(tiff_path, out_dir, box_threshold=0.25, text_threshold=0.2, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    img, geo, crs = load_image_and_transform(tiff_path)
    pil_img = Image.fromarray(img)

    dino_processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_MODEL).to(device)

    inputs = dino_processor(images=pil_img, text=TEXT_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)
    results = dino_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=box_threshold, text_threshold=text_threshold,
        target_sizes=[pil_img.size[::-1]],
    )[0]
    boxes = results["boxes"].cpu().numpy()
    print(f"Grounding DINO found {len(boxes)} candidate turf regions")

    if len(boxes) == 0:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
    else:
        sam_processor = SamProcessor.from_pretrained(SAM_MODEL)
        sam_model = SamModel.from_pretrained(SAM_MODEL).to(device)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        # SAM is run per-box in chunks to keep memory bounded on large images
        for i in range(0, len(boxes), 8):
            batch_boxes = [boxes[i:i + 8].tolist()]
            sam_inputs = sam_processor(pil_img, input_boxes=batch_boxes, return_tensors="pt")
            # MPS doesn't support float64, cast any float64 tensors to float32
            for k, v in sam_inputs.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.float64:
                    sam_inputs[k] = v.to(torch.float32)
            sam_inputs = sam_inputs.to(device)
            
            with torch.no_grad():
                sam_out = sam_model(**sam_inputs)
            masks = sam_processor.image_processor.post_process_masks(
                sam_out.pred_masks.cpu(), sam_inputs["original_sizes"].cpu(),
                sam_inputs["reshaped_input_sizes"].cpu(),
            )[0]
            # masks shape: (batch_size_of_boxes, num_masks, H, W)
            # take the highest-IoU mask per box and OR it into the accumulator
            best_idx = sam_out.iou_scores[0].argmax(dim=-1)
            best = masks[torch.arange(masks.shape[0]), best_idx].numpy()
            for m in best:
                mask |= m.astype(np.uint8)

    image_id = tiff_path.split("/")[-1].rsplit(".", 1)[0]
    cv2.imwrite(f"{out_dir}/{image_id}_zeroshot_mask.png", mask * 255)
    overlay = img.copy()
    overlay[mask == 1] = (overlay[mask == 1] * 0.35 + np.array([255, 165, 0]) * 0.65).astype(np.uint8)
    cv2.imwrite(f"{out_dir}/{image_id}_zeroshot_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"[{image_id}] zero-shot turf coverage: {100*mask.sum()/mask.size:.2f}% -> {out_dir}/{image_id}_zeroshot_mask.png")
    return mask


def ensemble_agreement(unet_mask_path, zeroshot_mask_path, out_path):
    """High-confidence consensus mask = pixels both models agree are turf."""
    m1 = cv2.imread(unet_mask_path, cv2.IMREAD_GRAYSCALE) > 127
    m2 = cv2.imread(zeroshot_mask_path, cv2.IMREAD_GRAYSCALE) > 127
    agree = (m1 & m2).astype(np.uint8) * 255
    disagree = (m1 ^ m2).astype(np.uint8) * 255
    cv2.imwrite(out_path.replace(".png", "_agree.png"), agree)
    cv2.imwrite(out_path.replace(".png", "_disagree.png"), disagree)
    iou = (m1 & m2).sum() / max((m1 | m2).sum(), 1)
    print(f"Model agreement IoU: {iou:.3f}")
    return iou


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_dir", default="../results/external_preds")
    args = ap.parse_args()
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    run_zero_shot(args.image, args.out_dir)
