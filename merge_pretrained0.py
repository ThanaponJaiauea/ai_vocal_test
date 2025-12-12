#!/usr/bin/env python3
"""
merge_pretrained.py

Merge RVC pretrained models directly from G and D checkpoint files.
This preserves the full model structure needed by RVC.
"""
import torch
import os
import logging
from collections import OrderedDict

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def merge_state_dicts(*state_dicts):
    """
    Merge multiple state dicts by averaging.
    All models get equal weight (1/N where N is number of models).
    """
    if len(state_dicts) == 0:
        raise ValueError("Need at least one state dict to merge")
    
    if len(state_dicts) == 1:
        return state_dicts[0]
    
    num_models = len(state_dicts)
    weight = 1.0 / num_models
    
    logging.info(f'Merging {num_models} models with equal weights ({weight:.3f} each)...')
    
    merged = OrderedDict()
    
    # Get all unique keys across all models
    all_keys = set()
    for sd in state_dicts:
        all_keys.update(sd.keys())
    
    # Get common keys (present in all models)
    common_keys = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        common_keys &= set(sd.keys())
    
    logging.info(f'Found {len(common_keys)} common parameters across all models')
    
    # Merge common keys
    for key in common_keys:
        tensors = [sd[key] for sd in state_dicts]
        
        # Check if all shapes match
        shapes = [t.shape for t in tensors]
        if len(set(shapes)) > 1:
            logging.warning(f'Shape mismatch for {key}: {shapes}. Using first model.')
            merged[key] = tensors[0]
        else:
            # Average all tensors
            merged[key] = sum(tensors) * weight
    
    # Handle keys not in all models
    only_in_some = all_keys - common_keys
    if only_in_some:
        logging.info(f'Found {len(only_in_some)} keys not in all models')
        for key in only_in_some:
            # Use from first model that has it
            for sd in state_dicts:
                if key in sd:
                    merged[key] = sd[key]
                    break
    
    return merged


def main():
    # ==========================================
    # ⚙️ CONFIGURATION: Add your models here
    # Format: ('Folder Path', Epoch_Number)
    # ==========================================
    models_config = [
        ('assets/model1', 35200),
        ('assets/model2', 74000),
        ('assets/model3', 27200),
        ('assets/model4', 96400),
        ('assets/model5', 36400),
        ('assets/model6', 50800),
    ]
    # ==========================================

    # Output directories
    os.makedirs('merge_G', exist_ok=True)
    os.makedirs('merge_D', exist_ok=True)
    
    g_state_dicts = []
    d_state_dicts = []
    
    logging.info('=' * 60)
    logging.info(f'PREPARING TO MERGE {len(models_config)} MODELS')
    logging.info('=' * 60)

    # Load all models
    for i, (folder, epoch) in enumerate(models_config):
        # Construct paths
        g_path = f"{folder}/G_{epoch}.pth"
        d_path = f"{folder}/D_{epoch}.pth"
        
        logging.info(f'[{i+1}/{len(models_config)}] Loading from: {folder} (Epoch {epoch})')
        
        # Load Generator
        if os.path.exists(g_path):
            logging.info(f'  - Loading G: {g_path}')
            g_ckpt = torch.load(g_path, map_location='cpu')
            g_state = g_ckpt if not isinstance(g_ckpt, dict) or 'model' not in g_ckpt else g_ckpt.get('model', g_ckpt)
            g_state_dicts.append(g_state)
        else:
            logging.error(f'  ❌ File not found: {g_path}')
            return

        # Load Discriminator
        if os.path.exists(d_path):
            logging.info(f'  - Loading D: {d_path}')
            d_ckpt = torch.load(d_path, map_location='cpu')
            d_state = d_ckpt if not isinstance(d_ckpt, dict) or 'model' not in d_ckpt else d_ckpt.get('model', d_ckpt)
            d_state_dicts.append(d_state)
        else:
            logging.error(f'  ❌ File not found: {d_path}')
            return

    # Merge Generators
    logging.info('=' * 60)
    logging.info('MERGING GENERATORS')
    logging.info('=' * 60)
    
    merged_g = merge_state_dicts(*g_state_dicts)
    
    # Save in RVC-compatible format with 'model' key
    output_g_path = 'merge_G/G.pth'
    output_g = {'model': merged_g}
    torch.save(output_g, output_g_path)
    logging.info(f'✓ Saved merged Generator to: {output_g_path}')
    
    # Merge Discriminators
    logging.info('=' * 60)
    logging.info('MERGING DISCRIMINATORS')
    logging.info('=' * 60)
    
    merged_d = merge_state_dicts(*d_state_dicts)
    
    # Save in RVC-compatible format with 'model' key
    output_d_path = 'merge_D/D.pth'
    output_d = {'model': merged_d}
    torch.save(output_d, output_d_path)
    logging.info(f'✓ Saved merged Discriminator to: {output_d_path}')
    
    logging.info('=' * 60)
    logging.info('MERGE COMPLETED SUCCESSFULLY!')
    logging.info('=' * 60)
    logging.info(f'Generator: {output_g_path}')
    logging.info(f'Discriminator: {output_d_path}')
    logging.info('')
    logging.info('These files should now be compatible with RVC WebUI.')


if __name__ == '__main__':
    main()
