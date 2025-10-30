"""
Utility script to inspect channel ordering and tetrode groups for the current dataset.

Usage:
    C:\\Users\\ryoi\\AppData\\Local\\anaconda3\\envs\\spikeinterface\\python.exe inspect_channels.py

It prints:
  • the raw channel IDs exposed by SpikeInterface
  • the channel IDs grouped per tetrode (using CHANNELS_PER_TETRODE from the pipeline)
  • a reminder of how to populate BAD_CHANNELS for a specific tetrode
"""

from pathlib import Path
import sys

import spikeinterface.extractors as se

ROOT_DIR_PATH = Path(__file__).resolve().parent.parent
PIPELINES_DIR = ROOT_DIR_PATH / "Pipelines"
if PIPELINES_DIR.exists():
    sys.path.insert(0, str(PIPELINES_DIR))

from spike_sorting_tetrodes_pipeline import (
    CHANNELS_PER_TETRODE,
    ROOT_DIR,
    SESSION_INDEX,
    SESSION_SELECTION,
    SESSION_SUBPATH,
    STREAM_NAME,
    discover_oe_stream_names,
    make_tetrode_groups,
    choose_recording_folder,
)


def main() -> None:
    data_path = choose_recording_folder(ROOT_DIR, SESSION_SUBPATH, SESSION_SELECTION, SESSION_INDEX)
    stream = STREAM_NAME or discover_oe_stream_names(data_path)[0]

    recording = se.read_openephys(data_path, stream_name=stream)
    print(f"Recording path: {data_path}")
    print(f"Using stream: {stream}")
    print(f"Total channels: {recording.get_num_channels()}")

    channel_ids = list(recording.channel_ids)
    print("\nChannel IDs (in acquisition order):")
    print(channel_ids)
    print("\nChannel indices -> IDs:")
    for idx, cid in enumerate(channel_ids):
        print(f"  {idx:02d}: {cid}")

    groups = make_tetrode_groups(channel_ids, CHANNELS_PER_TETRODE)
    print("\nTetrode groups:")
    for idx, group in enumerate(groups):
        indices = [channel_ids.index(ch) for ch in group]
        group_str = ", ".join(f"{name} (idx {index})" for name, index in zip(group, indices))
        print(f"  Tetrode {idx + 1:02d}: {group_str}")

    print(
        "\nPopulate BAD_CHANNELS with the identifiers you want to drop "
        "(strings like 'CH40' or integers, depending on channel_ids above)."
    )
    print("Example: BAD_CHANNELS = ['CH40', 'CH38']  # or BAD_CHANNELS = [40, 38]\n")


if __name__ == "__main__":
    main()
