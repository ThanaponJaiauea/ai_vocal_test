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
    # ⚙️ CONFIGURATION
    # ==========================================
    
    # 1. Base Model (ยืนพื้น 50%)
    # Folder containing f0G40k.pth and f0D40k.pth
    base_model_folder = 'assets/model_base_0'
    
    # 2. New Models to Merge (รวมกันแล้วเอาไปผสมกับ Base อีก 50%)
    # Format: ('Folder Path', Epoch_Number)
    new_models_config = [
       ('assets/model1', 35200),
       ('assets/model2', 74000),
       ('assets/model3', 27200),
       ('assets/model4', 96400),
       ('assets/model5', 36400),
    ]
    # ==========================================

    # Output directories
    os.makedirs('merge_G', exist_ok=True)
    os.makedirs('merge_D', exist_ok=True)
    
    logging.info('=' * 60)
    logging.info(f'MODE: Base Model (f0G/f0D) + {len(new_models_config)} New Models')
    logging.info('=' * 60)

    # --- Helper to load a model ---
    def load_checkpoint(path):
        if not os.path.exists(path):
            logging.error(f'❌ File not found: {path}')
            return None
        logging.info(f'  - Loading: {path}')
        ckpt = torch.load(path, map_location='cpu')
        return ckpt if not isinstance(ckpt, dict) or 'model' not in ckpt else ckpt.get('model', ckpt)

    # --- PROCESS GENERATORS ---
    logging.info('\n' + '=' * 30 + ' PROCESSING GENERATORS ' + '=' * 30)
    
    # 1. Load Base G (f0G40k.pth)
    base_g_path = f"{base_model_folder}/f0G40k.pth"
    logging.info(f'Loading Base Generator...')
    base_g = load_checkpoint(base_g_path)
    if base_g is None: return

    # 2. Load New Models G (G_xxxx.pth)
    new_g_list = []
    logging.info(f'Loading {len(new_models_config)} New Generators...')
    for folder, epoch in new_models_config:
        path = f"{folder}/G_{epoch}.pth"
        g = load_checkpoint(path)
        if g is None: return
        new_g_list.append(g)

    # 3. Merge New Models First
    logging.info('Step 1: Merging new models together...')
    merged_new_g = merge_state_dicts(*new_g_list)
    
    # 4. Merge with Base (50/50)
    logging.info('Step 2: Merging result with Base Model (50/50)...')
    final_g = merge_state_dicts(base_g, merged_new_g)
    
    # Save G
    output_g_path = 'merge_G/f0G40k.pth'
    torch.save({'model': final_g}, output_g_path)
    logging.info(f'✓ Saved Final Generator to: {output_g_path}')


    # --- PROCESS DISCRIMINATORS ---
    logging.info('\n' + '=' * 30 + ' PROCESSING DISCRIMINATORS ' + '=' * 30)
    
    # 1. Load Base D (f0D40k.pth)
    base_d_path = f"{base_model_folder}/f0D40k.pth"
    logging.info(f'Loading Base Discriminator...')
    base_d = load_checkpoint(base_d_path)
    if base_d is None: return

    # 2. Load New Models D (D_xxxx.pth)
    new_d_list = []
    logging.info(f'Loading {len(new_models_config)} New Discriminators...')
    for folder, epoch in new_models_config:
        path = f"{folder}/D_{epoch}.pth"
        d = load_checkpoint(path)
        if d is None: return
        new_d_list.append(d)

    # 3. Merge New Models First
    logging.info('Step 1: Merging new models together...')
    merged_new_d = merge_state_dicts(*new_d_list)
    
    # 4. Merge with Base (50/50)
    logging.info('Step 2: Merging result with Base Model (50/50)...')
    final_d = merge_state_dicts(base_d, merged_new_d)
    
    # Save D
    output_d_path = 'merge_D/f0D40k.pth'
    torch.save({'model': final_d}, output_d_path)
    logging.info(f'✓ Saved Final Discriminator to: {output_d_path}')
    
    logging.info('\n' + '=' * 60)
    logging.info('MERGE COMPLETED SUCCESSFULLY!')
    logging.info('=' * 60)
    logging.info(f'Generator: {output_g_path}')
    logging.info(f'Discriminator: {output_d_path}')



if __name__ == '__main__':
    main()
