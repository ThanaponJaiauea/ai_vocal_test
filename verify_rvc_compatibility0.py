#!/usr/bin/env python3
"""
verify_rvc_compatibility.py

Verify that the merged model files are compatible with RVC by checking
for all required keys that RVC expects.
"""
import torch

def check_generator(path):
    print(f"\n{'='*60}")
    print(f"Checking Generator: {path}")
    print('='*60)
    
    ckpt = torch.load(path, map_location='cpu')
    
    # Required prefixes for RVC v2 Generator
    required_prefixes = [
        'enc_p.',      # Text encoder
        'dec.',        # Decoder/Generator
        'enc_q.',      # Posterior encoder
        'flow.',       # Normalizing flow
        'emb_g.'       # Speaker embedding
    ]
    
    found_prefixes = set()
    if isinstance(ckpt, dict):
        for key in ckpt.keys():
            prefix = key.split('.')[0] + '.'
            found_prefixes.add(prefix)
    
    print(f"\nFound prefixes: {sorted(found_prefixes)}")
    print(f"\nRequired prefixes check:")
    all_found = True
    for prefix in required_prefixes:
        found = any(k.startswith(prefix) for k in ckpt.keys())
        status = "✓" if found else "✗"
        print(f"  {status} {prefix}")
        if not found:
            all_found = False
    
    if all_found:
        print(f"\n✅ Generator appears COMPATIBLE with RVC!")
    else:
        print(f"\n❌ Generator is MISSING required components!")
    
    return all_found


def check_discriminator(path):
    print(f"\n{'='*60}")
    print(f"Checking Discriminator: {path}")
    print('='*60)
    
    ckpt = torch.load(path, map_location='cpu')
    
    # Required prefixes for RVC v2 Discriminator
    required_prefixes = [
        'discriminators.'
    ]
    
    found_prefixes = set()
    if isinstance(ckpt, dict):
        for key in ckpt.keys():
            prefix = key.split('.')[0] + '.'
            found_prefixes.add(prefix)
    
    print(f"\nFound prefixes: {sorted(found_prefixes)}")
    print(f"\nRequired prefixes check:")
    all_found = True
    for prefix in required_prefixes:
        found = any(k.startswith(prefix) for k in ckpt.keys())
        status = "✓" if found else "✗"
        print(f"  {status} {prefix}")
        if not found:
            all_found = False
    
    if all_found:
        print(f"\n✅ Discriminator appears COMPATIBLE with RVC!")
    else:
        print(f"\n❌ Discriminator is MISSING required components!")
    
    return all_found


if __name__ == '__main__':
    g_ok = check_generator('merge_G/f0G40k.pth')
    d_ok = check_discriminator('merge_D/f0D40k.pth')
    
    print(f"\n{'='*60}")
    print("FINAL VERDICT")
    print('='*60)
    if g_ok and d_ok:
        print("✅ Both files should work with RVC WebUI!")
        print("\nTo use:")
        print("1. Copy merge_G/f0G40k.pth to RVC's assets/pretrained_v2/")
        print("2. Copy merge_D/f0D40k.pth to RVC's assets/pretrained_v2/")
        print("3. Use them as pretrained models in training")
    else:
        print("❌ Files may not be compatible. Check missing components above.")
    print('='*60 + '\n')
