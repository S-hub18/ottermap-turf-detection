"""
Runs the trained model on a full-size (unseen) GeoTIFF: tiles it the same
way as training, predicts each tile, stitches predictions back together
with overlap-averaging, and writes:
  - a binary turf mask (PNG, same pixel grid as input)
  - an overlay visualization (PNG)
  - prediction confidence map (NPZ, for downstream thresholding/QA)

This is the script the Ottermap team runs for evaluation:
    python inference.py --image input_image.tif --weights ../weights/turf_unet_resnet34.pt
or
    python inference.py --input ./images/ --weights ../weights/turf_unet_resnet34.pt
"""
import argparse
import glob
import os
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

from geoutils import load_image_and_transform


def get_infer_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def load_model(weights_path, device):
    ckpt = torch.load(weights_path, map_location=device)
    model = smp.Unet(encoder_name=ckpt["encoder"], encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"Loaded weights from {weights_path} (val_iou={ckpt.get('val_iou', 'n/a')})")
    return model


def sliding_window_predict(img, model, device, tile_size=512, stride=384, batch_size=8):
    h, w = img.shape[:2]
    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)
    img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    ph, pw = img_padded.shape[:2]

    prob_sum = np.zeros((ph, pw), dtype=np.float32)
    count = np.zeros((ph, pw), dtype=np.float32)
    transform = get_infer_transform()

    coords = []
    for top in range(0, ph - tile_size + 1, stride):
        for left in range(0, pw - tile_size + 1, stride):
            coords.append((top, left))
    # ensure right/bottom edges are covered
    if (ph - tile_size) % stride != 0:
        coords += [(ph - tile_size, left) for left in range(0, pw - tile_size + 1, stride)]
    if (pw - tile_size) % stride != 0:
        coords += [(top, pw - tile_size) for top in range(0, ph - tile_size + 1, stride)]

    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i:i + batch_size]
        batch_tiles = [img_padded[t:t+tile_size, l:l+tile_size] for t, l in batch_coords]
        batch_tensors = torch.stack([transform(image=t)["image"] for t in batch_tiles]).to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(batch_tensors)).cpu().numpy()[:, 0]
        for (t, l), p in zip(batch_coords, preds):
            prob_sum[t:t+tile_size, l:l+tile_size] += p
            count[t:t+tile_size, l:l+tile_size] += 1

    prob = prob_sum / np.maximum(count, 1)
    return prob[:h, :w]


def run_on_image(tiff_path, model, device, out_dir, tile_size, stride, thresh):
    image_id = os.path.splitext(os.path.basename(tiff_path))[0]
    img, geo, crs = load_image_and_transform(tiff_path)

    prob = sliding_window_predict(img, model, device, tile_size, stride)
    mask = (prob > thresh).astype(np.uint8)

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(f"{out_dir}/{image_id}_prob.npz", prob=prob)
    cv2.imwrite(f"{out_dir}/{image_id}_mask.png", mask * 255)

    overlay = img.copy()
    overlay[mask == 1] = (overlay[mask == 1] * 0.35 + np.array([0, 255, 0]) * 0.65).astype(np.uint8)
    cv2.imwrite(f"{out_dir}/{image_id}_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    coverage = 100 * mask.sum() / mask.size
    print(f"[{image_id}] predicted turf coverage: {coverage:.2f}%  -> {out_dir}/{image_id}_mask.png")
    return mask, geo, crs, image_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="single .tif/.tiff file")
    ap.add_argument("--input", help="directory of .tif/.tiff files")
    ap.add_argument("--weights", default="../weights/turf_unet_resnet34.pt")
    ap.add_argument("--out_dir", default="../results/external_preds")
    ap.add_argument("--tile_size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=384)
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    if not args.image and not args.input:
        raise SystemExit("Provide --image <file> or --input <directory>")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = load_model(args.weights, device)

    paths = [args.image] if args.image else []
    if args.input:
        paths = (glob.glob(os.path.join(args.input, "*.tif")) + 
                 glob.glob(os.path.join(args.input, "*.tiff")) +
                 glob.glob(os.path.join(args.input, "*.jpg")) +
                 glob.glob(os.path.join(args.input, "*.jpeg")))
    
    for p in paths:
        run_on_image(p, model, device, args.out_dir, args.tile_size, args.stride, args.thresh)
