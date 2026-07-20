import os
import json
import torch

def decode_tokens(token_indices, vocab, start_time, duration):
    idx_to_token = {v: k for k, v in vocab.items()}
    current_time = 0.0
    
    snippet = []
    
    for idx in token_indices:
        token = idx_to_token.get(idx, "<UNK>")
        
        if token.startswith("TIME_SHIFT_"):
            shift = float(token.split("_")[2])
            current_time += shift
            if start_time <= current_time <= start_time + duration:
                snippet.append(token)
                
        elif token.startswith("NOTE_"):
            if start_time <= current_time <= start_time + duration:
                snippet.append(token)
                
        if current_time > start_time + duration:
            break
            
    return snippet

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, "data", "processed")
    manifest_path = os.path.join(processed_dir, "manifest.json")
    vocab_path = os.path.join(processed_dir, "vocab.json")
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
        
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        
    worst_filename = None
    for fname in manifest.get("processed_files", {}):
        if "MIDI-Unprocessed_SMF_12_01_2004_01-05_ORIG_MID--AUDIO_12_R1_2004_03_Track03_wav--1.midi" in fname:
            worst_filename = fname
            break
            
    if not worst_filename:
        print("File not found in manifest!")
        return
        
    info = manifest["processed_files"][worst_filename]
    split = info["split"]
    shard_name = info["shard"]
    shard_path = os.path.join(processed_dir, "shards", shard_name)
    
    print(f"File: {worst_filename}")
    print(f"Split: {split}")
    print(f"Shard: {shard_name}")
    
    data = torch.load(shard_path, weights_only=False)
    tokens = data["tokens"].tolist()
    
    # decode around offset 2000 for 10 quarter notes
    snippet = decode_tokens(tokens, vocab, start_time=1995, duration=10)
    print("\n--- SNIPPET AROUND OFFSET 1995-2005 ---")
    for t in snippet[:30]:
        print(t)
    if len(snippet) > 30:
        print(f"... and {len(snippet)-30} more tokens in this window")

if __name__ == '__main__':
    main()
