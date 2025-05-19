import pyxdf
import argparse
import json # For pretty printing the info dictionary

def inspect_xdf(file_path):
    """
    Loads an XDF file and prints detailed information about each stream.
    """
    print(f"Attempting to load XDF file: {file_path}\n")
    try:
        streams, header = pyxdf.load_xdf(file_path)
    except Exception as e:
        print(f"Error loading XDF file: {e}")
        return

    print(f"Successfully loaded XDF file. Found {len(streams)} stream(s).\n")
    print("="*50)

    for i, stream in enumerate(streams):
        print(f"\n--- Stream {i+1} ---")
        
        # Pretty print the stream['info'] dictionary
        # We use json.dumps for a nicely formatted string output
        info_str = json.dumps(stream['info'], indent=4, sort_keys=True, default=str) # default=str handles non-serializable items
        print("Stream Info:")
        print(info_str)
        
        print(f"\nNumber of channels in time_series: {stream['time_series'].shape[1] if stream['time_series'].ndim > 1 else 1}")
        print(f"Number of samples in time_series: {len(stream['time_series'])}")
        print(f"Effective sampling rate (approx): {stream['info'].get('effective_srate', 'N/A')}")
        print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect streams within an XDF file.")
    parser.add_argument("xdf_file_path", type=str, help="Path to the .xdf file to inspect.")
    
    args = parser.parse_args()
    
    inspect_xdf(args.xdf_file_path)

    print("\nInspection complete. Use the 'Stream Info' details (especially 'name', 'type', ")
    print("and channel labels within 'desc') to configure your DataLoader in the pipeline.")