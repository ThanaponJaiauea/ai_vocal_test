<!-- @format -->

# ai_vocal_test

RVC Model Merging Tool - Merge multiple RVC pretrained models into a single combined model.

## Features

- Merge multiple RVC v2 Generator and Discriminator models
- Support for 2+ models with equal weight averaging
- Automatic RVC-compatible format (`{'model': state_dict}`)
- Verification tools to check model compatibility

## Files

- `merge_pretrained.py` - Main script to merge models
- `verify_rvc_compatibility.py` - Verify merged models are RVC-compatible

## Usage

1. Place your model checkpoints in separate folders (e.g., `model1/`, `model2/`, `model3/`)
2. Edit `merge_pretrained.py` to set the paths to your G and D checkpoint files
3. Run the merge:
   ```bash
   python merge_pretrained.py
   ```
4. Output files will be in `merge_G/f0G40k.pth` and `merge_D/f0D40k.pth`
5. Copy these to your RVC `assets/pretrained_v2/` directory

## Requirements

- Python 3.x
- PyTorch

## Notes

- Model files (`.pth`) are excluded from git due to large file sizes
- The merged models combine characteristics from all input models with equal weights
