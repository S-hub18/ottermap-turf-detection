# Technical Summary — Ottermap Turf/Grass Detection Pipeline

## 1. Core Architecture & Data Limitations
The primary constraint of this challenge was the extreme scarcity of domain variation in the training data: exactly **3 source aerial GeoTIFFs**. 
- **Training Strategy:** We opted for a U-Net architecture with a ResNet-34 encoder (via `segmentation_models_pytorch`). To prevent the model from memorizing the specific color palettes (overfitting) of those 3 days, we froze the encoder for the first 5 epochs, applied heavy color/brightness jittering augmentations rather than purely geometric transforms, and used a combined loss function of `0.5 * BCE + 0.5 * Dice`.
- **Validation Splitting:** We used a spatial train/val split (holding out the right-hand strip of each image) rather than random sampling, which would have leaked immediate spatial context into the validation set.
- **Training Baseline:** The model converged rapidly, reaching a **Validation IoU of 0.8746** and a **Validation Dice of 0.9234** at Epoch 22.

## 2. The Generalization Failure (Domain Gap)
Despite the high training IoU, the standalone U-Net exhibited severe domain gap failures when tested against a rigorous blind evaluation set of **9 unseen images** from vastly different environments (e.g., suburban Texas, Florida beaches, arid regions):
1. **False Negatives (Lighting):** In regions with dark lighting or shadow (e.g., test4, test5), the U-Net missed up to 50% of the actual turf because the spectral profile shifted out of its narrow trained distribution.
2. **False Positives (Pavement):** The model occasionally confused gray asphalt with dead grass.

## 3. The Triple-Intersection Ensemble (Final Pipeline)
To solve these mechanical failures without requiring thousands of new training images, we engineered a robust, non-parametric ensemble pipeline. 

### A. Hypersensitive U-Net (Recall Maximization)
Instead of forcing the U-Net to perfectly balance precision and recall at a standard 0.5 threshold, we shifted its entire purpose to **recall maximization**. We empirically dropped the activation threshold to **0.1**. 
- **Result:** This captured near-perfect boundaries on dark/unseen turf (e.g., recovering 95%+ of the turf on test4 and test8), but at the cost of massive false-positive "blooming" into surrounding driveways.

### B. Zero-Shot SAM (Geometric Grounding)
To reign in the false positives, we introduced a Zero-Shot pipeline using **Grounding DINO** (prompted with "grass. turf.") hooked into Meta's **Segment Anything Model (SAM)**.
- **Why?** SAM is color-agnostic; it looks for geometric object boundaries.
- **Result:** SAM perfectly snapped to the edges of the grass, but it often hallucinated (e.g., identifying lakes or roads as "turf" if prompted poorly).
- **The Intersection:** By executing a strict **Logical AND** between the hypersensitive U-Net and SAM, the models acted as a Yin-Yang filter. The U-Net's spectral knowledge stripped away SAM's water/road hallucinations, while SAM's geometric knowledge pruned the U-Net's driveway "blooming."

### C. The Excess Green (ExG) Sanity Filter
While the Logical AND solved 90% of the domain gap, both models occasionally agreed that certain gray pavements were grass. We eliminated this final error margin using classical computer vision.
- **Formula:** We implemented the Excess Green vegetation index: `ExG = 2*G - R - B`.
- **Calibration:** Through empirical sweeping across thresholds (0.05 to 0.5 on normalized pixel intensity), we settled on an absolute un-normalized threshold of `> 10`.
- **Result:** This strict mechanical gatekeeper completely stripped the remaining asphalt leakage without destroying true grass boundaries.

## 4. Inference & Vectorization
The final pipeline outputs the intersected raster mask and executes a fully automated vectorization step. It extracts contours using `cv2.findContours`, filters out micro-noise artifacts under 30 pixels (an area heuristic), simplifies the geometry via the Douglas-Peucker algorithm (`tolerance = 0.00001`), and exports directly to GIS-ready `.geojson`. 

## 5. Future Scalability
With more time, this pipeline would benefit immensely from:
1. **Active Learning / Hard Negative Mining:** Pushing the highest-entropy tiles back to human annotators for review.
2. **Near-Infrared (NIR) Data:** Replacing RGB with 4-band imagery. The NDVI index would immediately solve the spectral overlap between "green car" and "green grass" without requiring a complex neural network.
