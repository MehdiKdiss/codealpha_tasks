import os
import json
import torch
from torch.utils.data import Dataset

class MaestroDataset(Dataset):
    """
    Lazy-loading Dataset for MAESTRO token streams.
    
    Instead of pre-materializing millions of sliding windows on disk (which would
    consume gigabytes of storage), we load the 1D token streams into memory 
    (which only takes ~40MB total) and slice out windows on-the-fly.
    """
    def __init__(self, base_dir, split="train", seq_len=100, stride=1):
        self.seq_len = seq_len
        self.stride = stride
        
        manifest_path = os.path.join(base_dir, "data", "processed", "manifest.json")
        shards_dir = os.path.join(base_dir, "data", "processed", "shards")
        
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            
        self.file_tokens = []
        self.index_map = []  # Maps global dataset idx -> (file_idx, token_start_idx)
        
        # Load all shards for the requested split into memory
        file_idx = 0
        for fname, info in manifest.get("processed_files", {}).items():
            if info["split"] == split:
                shard_path = os.path.join(shards_dir, info["shard"])
                
                # Load the 1D tensor of tokens
                data = torch.load(shard_path, weights_only=False)
                # Convert to long immediately for nn.Embedding
                tokens = data["tokens"].to(torch.long)
                
                num_tokens = len(tokens)
                # We need at least seq_len + 1 tokens to form one (input, target) pair
                if num_tokens > seq_len:
                    self.file_tokens.append(tokens)
                    
                    # Calculate how many windows we can extract from this file
                    # We use seq_len to ensure we have a full input sequence,
                    # and +1 to ensure we have the shifted targets.
                    num_windows = (num_tokens - seq_len) // stride
                    
                    for i in range(num_windows):
                        start_idx = i * stride
                        self.index_map.append((file_idx, start_idx))
                    
                    file_idx += 1
                    
        self.length = len(self.index_map)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        file_idx, start_idx = self.index_map[idx]
        tokens = self.file_tokens[file_idx]
        
        # We extract input sequence of length seq_len
        x = tokens[start_idx : start_idx + self.seq_len]
        
        # We extract target sequence shifted by 1 (many-to-many training)
        # This matches the model's output shape of (batch, seq_len, vocab_size)
        # and is massively more efficient than predicting just a single target token.
        y = tokens[start_idx + 1 : start_idx + self.seq_len + 1]
        
        return x, y
