import pyxdf
import argparse
import json # For pretty printing the info dictionary
import numpy as np # Import numpy to check array type

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
        
        # Print just the stream name
        info_str = f"Stream Name: {stream['info'].get('name', ['N/A'])[0]}" # Get the first element if 'name' is a list
        print("Stream Info:")
        print(info_str)
        
        # Check if time_series is a numpy array before accessing shape/ndim
        if isinstance(stream['time_series'], np.ndarray):
            print(f"\nNumber of channels in time_series: {stream['time_series'].shape[1] if stream['time_series'].ndim > 1 else 1}")
        else:
            print("\nTime series data is not a standard NumPy array (likely markers or strings).")
            print("Number of channels concept may not apply in the same way.")
        print(f"Number of samples in time_series: {len(stream['time_series'])}")
        print(f"Effective sampling rate (approx): {stream['info'].get('effective_srate', 'N/A')}")
        print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect streams within an XDF file.")
    # Make the argument optional and set the default to the specified pilot file path
    parser.add_argument("--xdf_file_path", type=str,
                        default=r"D:\pilotRawData\EV_P005\EV_P005.xdf",
                        help="Path to the .xdf file to inspect (defaults to a pilot file if not specified).")
    
    args = parser.parse_args()
    
    inspect_xdf(args.xdf_file_path)

    print("\nInspection complete. Use the 'Stream Info' details (especially 'name', 'type', ")
    print("and channel labels within 'desc') to configure your DataLoader in the pipeline.")