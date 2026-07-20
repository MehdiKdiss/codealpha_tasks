import os
import csv
import pretty_midi
from tqdm import tqdm

def main():
    csv_path = os.path.join("CodeAlpha_MusicGenerationAI", "data", "raw", "maestro-v3.0.0.csv")
    midi_base_dir = os.path.join("CodeAlpha_MusicGenerationAI", "data", "raw", "maestro-v3.0.0-midi", "maestro-v3.0.0")
    
    print(f"Reading CSV metadata from: {csv_path}")
    print(f"Expecting MIDI files in: {midi_base_dir}")
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
        
    if not os.path.exists(midi_base_dir):
        print(f"Error: MIDI base directory not found at {midi_base_dir}")
        return
        
    # Official expected counts
    EXPECTED_TOTAL = 1276
    EXPECTED_SPLITS = {
        'train': 962,
        'validation': 137,
        'test': 177
    }
    
    records = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
            
    total_records = len(records)
    print(f"Found {total_records} records in the CSV file (Expected: {EXPECTED_TOTAL}).")
    
    # Track stats
    split_counts = {'train': 0, 'validation': 0, 'test': 0}
    missing_files = []
    corrupt_files = []
    zero_byte_files = []
    success_count = 0
    
    print("Verifying MIDI files readability and integrity...")
    # Use tqdm to show progress since we are reading ~1,276 files
    for record in tqdm(records, desc="Checking MIDI files"):
        midi_filename = record['midi_filename']
        split = record['split']
        
        # Increment split count
        if split in split_counts:
            split_counts[split] += 1
        else:
            split_counts[split] = 1
            
        full_path = os.path.join(midi_base_dir, midi_filename.replace('/', os.sep))
        
        # Check existence
        if not os.path.exists(full_path):
            missing_files.append((midi_filename, split, "Missing"))
            continue
            
        # Check size
        file_size = os.path.getsize(full_path)
        if file_size == 0:
            zero_byte_files.append((midi_filename, split, "Zero-byte"))
            continue
            
        # Check readability using pretty_midi
        try:
            # We just load the midi object to ensure it parses without exceptions
            pm = pretty_midi.PrettyMIDI(full_path)
            success_count += 1
        except Exception as e:
            corrupt_files.append((midi_filename, split, f"Error: {str(e)}"))
            
    print("\n" + "="*50)
    print("         MAESTRO V3.0.0 VERIFICATION REPORT")
    print("="*50)
    print(f"Total CSV records parsed:      {total_records}")
    print(f"Total MIDI files verified OK:  {success_count}")
    
    # Split verification
    print("\n--- Split Verification ---")
    splits_match = True
    for split_name, expected in EXPECTED_SPLITS.items():
        actual = split_counts.get(split_name, 0)
        status = "OK" if actual == expected else "MISMATCH"
        if actual != expected:
            splits_match = False
        print(f"  {split_name.capitalize():<12}: Actual = {actual:<4} (Expected = {expected:<4}) -> {status}")
        
    if splits_match and total_records == EXPECTED_TOTAL:
        print("  Split counts matches the official MAESTRO v3.0.0 split specifications.")
    else:
        print("  WARNING: Split counts or total records do NOT match official specifications!")
        
    # File integrity verification
    print("\n--- File Integrity ---")
    print(f"  Missing files:   {len(missing_files)}")
    print(f"  Zero-byte files: {len(zero_byte_files)}")
    print(f"  Corrupt files:   {len(corrupt_files)}")
    
    if len(missing_files) > 0:
        print("\nMissing Files List (first 10 shown):")
        for f_name, spl, err in missing_files[:10]:
            print(f"  [{spl}] {f_name}")
            
    if len(zero_byte_files) > 0:
        print("\nZero-byte Files List (first 10 shown):")
        for f_name, spl, err in zero_byte_files[:10]:
            print(f"  [{spl}] {f_name}")
            
    if len(corrupt_files) > 0:
        print("\nCorrupt Files List (first 10 shown):")
        for f_name, spl, err in corrupt_files[:10]:
            print(f"  [{spl}] {f_name} - {err}")
            
    if len(missing_files) == 0 and len(zero_byte_files) == 0 and len(corrupt_files) == 0:
        print("  All files are present, non-empty, and successfully loaded using pretty_midi (no corruption found).")
        
    print("="*50)

if __name__ == "__main__":
    main()
