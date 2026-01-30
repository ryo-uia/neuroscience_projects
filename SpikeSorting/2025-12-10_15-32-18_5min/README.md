# 5?min sample (Open Ephys export)

Files in this folder:
- `continuous.dat` (int16, 64 channels, 30 kHz)
- `metadata.json` (dtype, sampling rate, channel count, frames, duration)

Adjust the path to this folder on your machine before running.

## SpikeInterface

```python
import spikeinterface as si
from pathlib import Path

folder = Path(r"C:\\path\\to\\2025-12-10_15-32-18_5min")
rec = si.read_binary(folder / "continuous.dat", sampling_frequency=30000, num_chan=64, dtype="int16")
print(rec)
```

## NumPy

```python
import numpy as np
from pathlib import Path
import json

folder = Path(r"C:\\path\\to\\2025-12-10_15-32-18_5min")
meta = json.loads((folder / "metadata.json").read_text())
data = np.memmap(folder / "continuous.dat", dtype=meta["dtype"], mode="r")
data = data.reshape(meta["num_frames"], meta["num_channels"])  # (time, channels)
print(data.shape)
```


## Channel labels

``
CH40
CH38
CH36
CH34
CH48
CH46
CH44
CH42
CH56
CH54
CH52
CH50
CH58
CH64
CH62
CH60
CH63
CH61
CH59
CH57
CH55
CH53
CH51
CH49
CH47
CH45
CH43
CH41
CH39
CH37
CH35
CH33
CH25
CH27
CH29
CH31
CH17
CH19
CH21
CH23
CH9
CH11
CH13
CH15
CH1
CH3
CH5
CH7
CH4
CH6
CH8
CH2
CH10
CH12
CH14
CH16
CH18
CH20
CH22
CH24
CH26
CH28
CH30
CH32
``

## Using this with the SpikeSorting pipeline

The pipeline currently expects an Open Ephys recording folder. For this 5?min binary clip, load it with `si.read_binary(...)` (see snippet above), or add a small loader shim that takes a `--binary-dat` path.
