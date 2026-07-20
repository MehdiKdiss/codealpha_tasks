import os
import json
import torch
from torch.utils.data import DataLoader

from src.dataset import MaestroDataset
from src.model import MusicLSTM

def main():
    print("=== Phase 3 Sanity Check ===")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Check Vocab Size
    vocab_path = os.path.join(base_dir, "data", "processed", "vocab.json")
    if not os.path.exists(vocab_path):
        print("ERROR: vocab.json not found! Run preprocessing first.")
        return
        
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")
    
    # 2. Instantiate Dataset and DataLoader
    print("\nInstantiating MaestroDataset...")
    seq_len = 100
    dataset = MaestroDataset(base_dir=base_dir, split="validation", seq_len=seq_len, stride=10)
    print(f"Validation dataset size (with stride=10): {len(dataset)}")
    
    if len(dataset) == 0:
        print("ERROR: Dataset is empty.")
        return
        
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 3. Load one batch
    x, y = next(iter(dataloader))
    print("\n--- Batch Information ---")
    print(f"Input shape (x): {x.shape} (Expected: {batch_size}, {seq_len})")
    print(f"Target shape (y): {y.shape} (Expected: {batch_size}, {seq_len})")
    print(f"Input dtype: {x.dtype}")
    print(f"Target dtype: {y.dtype}")
    
    if x.dtype != torch.long:
        print("WARNING: Input is not torch.long, this will crash the embedding layer!")
        
    # 4. Instantiate Model
    print("\nInstantiating MusicLSTM...")
    model = MusicLSTM(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_size=256,
        num_layers=2,
        dropout=0.2
    )
    print(model)
    
    # 5. Forward Pass
    print("\n--- Forward Pass ---")
    try:
        logits, hidden = model(x)
        print(f"Output shape (logits): {logits.shape} (Expected: {batch_size}, {seq_len}, {vocab_size})")
        print(f"Hidden state type: {type(hidden)}, Length: {len(hidden)}")
        print(f"Hidden state h_n shape: {hidden[0].shape} (Expected: 2, {batch_size}, 256)")
        print("\nSUCCESS! Phase 3 Implementation is structurally sound.")
    except Exception as e:
        print(f"FORWARD PASS FAILED: {str(e)}")

if __name__ == "__main__":
    main()
