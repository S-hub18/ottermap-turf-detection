import argparse
import subprocess
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="End-to-End Ottermap Turf Detection Pipeline")
    parser.add_argument("--image", required=True, help="Path to input image (.tif, .png, .jpg)")
    parser.add_argument("--weights", default="weights/turf_unet_resnet34_full.pt", help="Path to U-Net weights")
    parser.add_argument("--out_dir", default="results/pipeline_output", help="Directory to save final outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.image))[0]

    print(f"\n[1/4] Running Zero-Shot SAM Geometry Extraction...")
    sam_out_dir = os.path.join(args.out_dir, "temp_sam")
    sam_mask_path = os.path.join(sam_out_dir, f"{base_name}_zeroshot_mask.png")
    subprocess.run([
        sys.executable, "src/zero_shot_sam.py",
        "--image", args.image,
        "--out_dir", sam_out_dir,
        "--box_thresh", "0.1",
        "--text_thresh", "0.1"
    ], check=True)

    print(f"\n[2/4] Running Hypersensitive U-Net (Threshold 0.1)...")
    unet_out_dir = os.path.join(args.out_dir, "temp_unet")
    unet_mask_path = os.path.join(unet_out_dir, f"{base_name}_mask.png")
    subprocess.run([
        sys.executable, "src/inference.py",
        "--image", args.image,
        "--weights", args.weights,
        "--thresh", "0.1",
        "--out_dir", unet_out_dir
    ], check=True)

    print(f"\n[3/4] Running Triple-Intersection Ensemble (U-Net + SAM + ExG)...")
    final_mask_path = os.path.join(args.out_dir, f"{base_name}_final_mask.png")
    subprocess.run([
        sys.executable, "src/ensemble.py",
        "--unet_mask", unet_mask_path,
        "--sam_mask", sam_mask_path,
        "--image", args.image,
        "--out_mask", final_mask_path,
        "--exg_thresh", "10"
    ], check=True)

    print(f"\n[4/4] Vectorizing Final Mask to GIS Formats...")
    subprocess.run([
        sys.executable, "src/vectorize.py",
        "--image", args.image,
        "--mask", final_mask_path,
        "--out_dir", args.out_dir,
        "--min_area_px", "30",
        "--simplify_tol", "0.00001"
    ], check=True)

    print(f"\n✅ Pipeline Complete! All outputs saved to: {args.out_dir}")
    print(f" - Final Mask: {final_mask_path}")
    print(f" - Vector Data: {os.path.join(args.out_dir, base_name + '_predictions.geojson')}")

if __name__ == "__main__":
    main()
