import os
import glob
import music21
import mido
import csv
from collections import defaultdict
from tqdm import tqdm

DURATION_RESOLUTION = 0.25

def quantize_duration(dur, resolution=DURATION_RESOLUTION):
    if dur <= 0:
        return 0.0
    ticks = max(1, round(dur / resolution))
    return round(ticks * resolution, 4)

def check_midi(midi_path):
    warnings = 0
    total_notes = 0
    
    # We use mido for speed instead of music21.
    # We must calculate absolute time in quarter notes (beats).
    try:
        mid = mido.MidiFile(midi_path)
    except Exception:
        return 0, 0
    
    ticks_per_beat = mid.ticks_per_beat
    
    # Extract note onsets in ticks
    # To properly simulate music21, we need absolute time in quarter lengths.
    # music21 calculates offset based on tempo changes?
    # Actually, in MAESTRO, tempo is usually constant or mido can convert to seconds, 
    # but music21 quarterLength is just based on ticks and ticks_per_beat if it's purely MIDI!
    # Wait, music21 handles tempo, but offset is always in quarter notes (beats) based on ticks, ignoring tempo!
    
    # Let's collect all note onsets in absolute ticks
    abs_ticks = 0
    notes = []
    for track in mid.tracks:
        track_ticks = 0
        for msg in track:
            track_ticks += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append(track_ticks)
                
    # Convert to quarter notes
    events = [ (t / ticks_per_beat) for t in notes ]
    events.sort()
    
    total_notes = len(events)
    
    current_time = 0.0
    for offset in events:
        time_shift = quantize_duration(offset - current_time)
        if time_shift > 0:
            current_time += time_shift
        elif offset - current_time < -1e-6: # Using a small epsilon for floating point issues
            warnings += 1
            
    return warnings, total_notes

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "data", "raw", "maestro-v3.0.0-midi", "maestro-v3.0.0")
    csv_path = os.path.join(base_dir, "data", "raw", "maestro-v3.0.0.csv")
    
    # read csv
    files = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            files.append(row)
            
    total_warnings = 0
    affected_files = 0
    
    file_stats = []
    
    for row in tqdm(files):
        midi_path = os.path.join(raw_dir, row['midi_filename'].replace('/', os.sep))
        warns, notes = check_midi(midi_path)
        
        if warns > 0:
            total_warnings += warns
            affected_files += 1
            pct = (warns / notes * 100) if notes > 0 else 0
            
            file_stats.append({
                'filename': row['midi_filename'],
                'warnings': warns,
                'notes': notes,
                'pct': pct
            })
            
    # Sort by warnings descending
    file_stats.sort(key=lambda x: x['warnings'], reverse=True)
                
    print("\n--- RESULTS ---")
    print(f"Total distinct files affected: {affected_files}")
    print(f"Total warning count: {total_warnings}")
    print("\n--- TOP 5 MOST-AFFECTED FILES ---")
    for i, stats in enumerate(file_stats[:5], 1):
        print(f"{i}. {stats['filename']}")
        print(f"   Warnings: {stats['warnings']} / {stats['notes']} notes ({stats['pct']:.2f}%)")

if __name__ == '__main__':
    main()
