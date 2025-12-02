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
    # Paths to original checkpoint files
    model1_g = 'model1/G_400.pth'
    model1_d = 'model1/D_400.pth'

    model2_g = 'model2/G_400.pth'
    model2_d = 'model2/D_400.pth'

    model3_g = 'model3/G_450.pth'
    model3_d = 'model3/D_450.pth'
    
    # Output directories
    os.makedirs('merge_G', exist_ok=True)
    os.makedirs('merge_D', exist_ok=True)
    
    # Merge Generators
    logging.info('=' * 60)
    logging.info('MERGING GENERATORS')
    logging.info('=' * 60)
    
    logging.info(f'Loading: {model1_g}')
    g1_ckpt = torch.load(model1_g, map_location='cpu')
    
    logging.info(f'Loading: {model2_g}')
    g2_ckpt = torch.load(model2_g, map_location='cpu')
    
    logging.info(f'Loading: {model3_g}')
    g3_ckpt = torch.load(model3_g, map_location='cpu')
    
    # Extract state dicts (these are already full model state dicts)
    g1_state = g1_ckpt if not isinstance(g1_ckpt, dict) or 'model' not in g1_ckpt else g1_ckpt.get('model', g1_ckpt)
    g2_state = g2_ckpt if not isinstance(g2_ckpt, dict) or 'model' not in g2_ckpt else g2_ckpt.get('model', g2_ckpt)
    g3_state = g3_ckpt if not isinstance(g3_ckpt, dict) or 'model' not in g3_ckpt else g3_ckpt.get('model', g3_ckpt)
    
    merged_g = merge_state_dicts(g1_state, g2_state, g3_state)
    
    # Save in RVC-compatible format with 'model' key
    output_g_path = 'merge_G/f0G40k.pth'
    output_g = {'model': merged_g}
    torch.save(output_g, output_g_path)
    logging.info(f'✓ Saved merged Generator to: {output_g_path}')
    
    # Merge Discriminators
    logging.info('=' * 60)
    logging.info('MERGING DISCRIMINATORS')
    logging.info('=' * 60)
    
    logging.info(f'Loading: {model1_d}')
    d1_ckpt = torch.load(model1_d, map_location='cpu')
    
    logging.info(f'Loading: {model2_d}')
    d2_ckpt = torch.load(model2_d, map_location='cpu')
    
    logging.info(f'Loading: {model3_d}')
    d3_ckpt = torch.load(model3_d, map_location='cpu')
    
    # Extract state dicts
    d1_state = d1_ckpt if not isinstance(d1_ckpt, dict) or 'model' not in d1_ckpt else d1_ckpt.get('model', d1_ckpt)
    d2_state = d2_ckpt if not isinstance(d2_ckpt, dict) or 'model' not in d2_ckpt else d2_ckpt.get('model', d2_ckpt)
    d3_state = d3_ckpt if not isinstance(d3_ckpt, dict) or 'model' not in d3_ckpt else d3_ckpt.get('model', d3_ckpt)
    
    merged_d = merge_state_dicts(d1_state, d2_state, d3_state)
    
    # Save in RVC-compatible format with 'model' key
    output_d_path = 'merge_D/f0D40k.pth'
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
