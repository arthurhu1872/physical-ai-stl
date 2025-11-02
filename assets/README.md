# assets/

Place small generated demo assets here (e.g., 2D heat frames for MoonLight).
This directory is **.gitignored** for large files—commit only tiny samples.

## Generate 2D heat frames
```bash
python scripts/gen_heat2d_frames.py --nx 32 --ny 32 --nt 50 --outdir assets/heat2d_scalar
```
