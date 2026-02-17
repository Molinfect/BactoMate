# Training troubleshooting: head drift, 0 cells, background detected

When you fine-tune on top of **bact_phase_omni** (or another pretrained model), you may see:

1. **100 and 200 epochs find cells; 300 and 400 find no cells; 500 finds some again** – no obvious logic.
2. **Background is detected as cells** – the model predicts cells where there is only background in your masks.

This document explains why this happens and what to do.

---

## Why do 100/200 work but 300/400 give 0 cells (and 500 works again)?

This is **output head drift** from fine-tuning with a high learning rate and many epochs.

- Omnipose’s last layer (“output head”) produces a **distance / cell-probability** map. At inference, pixels with **distance > 0** (and above thresholds) become cells. The pretrained model has a certain output scale and bias.
- When you fine-tune with **many epochs** (e.g. 300, 400, 500) and **LR 0.01**, the optimizer keeps updating that head. The head can **drift**:
  - **100, 200**: Weights are still close to the pretrained values → output range still valid → **cells are found**.
  - **300, 400**: The head has drifted (e.g. bias goes negative, weight scale shrinks). The same “distance” channel no longer crosses the inference thresholds → **0 cells**.
  - **500**: Further updates happen to move the head again; by chance it can drift back into a range where some detections occur, or overfit in a way that partially generalizes → **some cells again**.

So the behaviour is **not random** – it comes from the head drifting in and out of a “valid” output range as training continues.

**What to do:**

- **Use an earlier checkpoint** that works: e.g. **epoch 100 or 200** for inference instead of 300/400/500.
- For **future training**: use **fewer epochs** (e.g. **50–100**) and/or a **lower learning rate** (e.g. **0.005**). Then run **Check pipeline** (compare checkpoints) to see the “Output head verdict”; prefer checkpoints that say “likely segments”.
- In the GUI: **Save every N epochs** (e.g. 50) so you can compare and pick the best checkpoint instead of relying only on the final one.

---

## Why is background detected as cells?

If your **masks** are correct (instance labels: background=0, each cell=1, 2, 3, …) and you still see **background predicted as cells**, common causes are:

1. **Head drift in the “other” direction**  
   With high LR and many epochs, the head can drift so that the **distance/cell channel** is too high everywhere – including in background. So “background” gets a positive distance and is turned into cells. Same mechanism as 0 cells, but the head moved the opposite way.

2. **Learning rate too high for fine-tuning**  
   **0.01** is already at the upper end for transfer learning. **0.005** or lower is safer. High LR can push the head into a regime where background and cell scores are wrong (either 0 cells or “everything is cell”).

3. **Mask format**  
   Omnipose expects **instance segmentation**: background=0, each cell a **unique** integer (1, 2, 3, …). If you used **semantic** masks (e.g. all cells = 1), the loss does not match what Omnipose assumes and the model can learn to predict “cell” in the wrong places. Double-check that masks are instance labels.

**What to do:**

- Use **fewer epochs** and **lower LR** (e.g. 50–100 epochs, LR 0.005).
- Use the **checkpoint that segments well** (e.g. 100 or 200) and avoid the final one if it over-segments.
- Run **Check pipeline** on your checkpoints to inspect the output head (bias, weight max). If the head is “weak” or extreme, try another checkpoint or retrain with lower LR/fewer epochs.
- Confirm **mask format**: background=0, one unique ID per cell.

---

## Quick reference

| Symptom | Likely cause | Action |
|--------|----------------|--------|
| 100/200 find cells, 300/400 don’t, 500 finds some again | Output head drift (many epochs + LR) | Use epoch 100 or 200 for inference; next time train with fewer epochs (50–100) and/or LR 0.005 |
| Background detected as cells | Head drifted the other way; or LR too high; or mask format | Lower LR; fewer epochs; use earlier checkpoint; verify instance masks (0=bg, 1,2,3,…=cells) |
| 0 cells at all checkpoints | Head too weak (bias/weights) | Run Check pipeline; try mask threshold -1 or -2; retrain with 50 epochs, LR 0.01 or 0.005 |

See also: **Check pipeline** (Compare two models) and the **Epochs** / **Learning rate** (?) tooltips in the Train model dialog.

**Full diagnostic (flat distance/flow, 0 cells):** If you need to check in all directions — checkpoint format, forward-pass output stats, training data format, and what BactiMate does vs Omnipose — run:

```bash
python -m bactimate.scripts.diagnose_flat_model omnipose_user_models/YOUR_MODEL.pth [--image path/to/image.tif] [--training-dir /path/to/train] [--compare-builtin bact_phase_omni]
```

This reports: (1) checkpoint keys and output-head bias/weights, (2) per-channel flow/distance stats from one eval, (3) training data shape and value ranges if you pass `--training-dir`, (4) that BactiMate does not change the architecture, and (5) concrete recommendations.

---

## How is the “valid range” determined so that cells are found?

Two places define what “valid” means:

### 1. At inference (Omnipose)

The model’s last layer outputs a **distance** (cell-probability) map. Omnipose/cellpose then applies a **mask threshold** (pixels with distance **> threshold** become cell pixels). BactiMate does **not** patch this logic (the only Omnipose-related patch is for `get_masks` when the distance array is `None` in frozen builds; it does not change the threshold). BactiMate only exposes a “Mask threshold” control in the UI (default **0**) and passes that value through to `model.eval(mask_threshold=...)`. The comparison “distance > mask_threshold” is done inside Omnipose/cellpose. So:

- Pixels with **distance > mask_threshold** are treated as **cell** pixels.
- The rest are **background**.
- Flow and other outputs are then used to split cell pixels into individual instances.

So the “range” at inference is: **distance must be above the threshold** (default 0). If the head drifts so that the distance map is mostly **below** 0, almost no pixels pass → **0 cells**. If it drifts so that distance is **above** 0 almost everywhere (including in real background), you get **background detected as cells**. The threshold itself is fixed (e.g. 0); it’s the **model output range** that moves with training. BactiMate can retry with **mask_threshold = -1 or -2** for user models when 0 cells are found, to see if slightly negative thresholds recover detections.

### 2. In Check pipeline (verdict)

The “Output head verdict” does **not** run a full forward pass on an image. It only inspects the **last layer’s parameters** in the checkpoint: `output.0.weight` and `output.0.bias`. That layer computes something like `distance = weight @ features + bias`. For the network to produce **positive** distance values (so pixels pass the default “distance > 0” rule), the combination of features and this layer must yield positive values. Empirically, BactiMate uses:

- **output.0.bias mean ≥ 0** → verdict **“likely segments”**: the layer adds a non‑negative offset, so outputs tend to be positive enough.
- **output.0.weight max < 0.2** → verdict **“likely 0 cells”**: the weights have shrunk so much that typical activations give small or negative outputs; the “0.2” is an empirical cutoff from models that segment vs models that don’t.
- **Bias < 0 but weight max ≥ 0.2** → **“may segment”**: might still get cells, but often only with **mask_threshold = -1 or -2**.

So the “range” in the verdict is **heuristic**: it’s not the exact inference threshold, but a simple check on the last layer’s scale and bias that has been found to correlate with “model finds cells” vs “model gives 0 cells”. The real test is always: run inference (e.g. with Check pipeline’s inference step or the Processing tab) and see how many objects you get; the verdict is a quick sanity check on the checkpoint.
