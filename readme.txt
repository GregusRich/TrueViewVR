This is a 2D → Stereo (VR) Proof of Concept

This is a **minimal working prototype** for turning a single 2D frame into a **stereoscopic pair** (Right + Left views).
It implements your *draft-then-constrain* idea in two stages:

1. **Geometry draft with DIBR**
   - Estimate depth from the right-eye image (monocular depth).
   - Convert depth → per-pixel disparity → warp to synthesize a *left-eye draft*.
   - Produce a *hole mask* marking disocclusions.

2. **(Optional) Depth-conditioned cleanup**
   - Use **Stable Diffusion + ControlNet-Depth** to clean the left draft **while obeying the left-eye depth map**.
   - This is where your “guided inpaint” comes in. You can keep as much of the warped pixels as possible and only regenerate the holes.

> Start with Stage 1 alone (no diffusion). If you have a GPU and want to push quality, enable Stage 2.

---

## Quick Start (local, Linux/macOS/Windows with CUDA recommended)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# run on a single image
python 2dto3d_poc.py --input /path/to/frame.png