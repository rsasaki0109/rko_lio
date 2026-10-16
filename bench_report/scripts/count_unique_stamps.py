"""Count points and unique per-point timestamps in the first N lidar messages of a rosbag2."""
import sys
from collections import Counter

import numpy as np
from rosbags.highlevel import AnyReader
from pathlib import Path
from rosbags.typesys import Stores, get_typestore

bag, topic, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
with AnyReader([Path(bag)], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
    conns = [c for c in reader.connections if c.topic == topic]
    for i, (conn, t, raw) in enumerate(reader.messages(connections=conns)):
        msg = reader.deserialize(raw, conn.msgtype)
        fields = {f.name: (f.offset, f.datatype) for f in msg.fields}
        name = 'timestamp' if 'timestamp' in fields else ('time' if 'time' in fields else ('t' if 't' in fields else None))
        off, dt = fields[name]
        dtype = {8: np.float64, 7: np.float32, 6: np.uint32}[dt]
        data = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(-1, msg.point_step)
        ts = data[:, off:off + np.dtype(dtype).itemsize].copy().view(dtype).ravel()
        runs = 1 + int(np.count_nonzero(ts[1:] != ts[:-1]))
        print(f"msg {i}: points={len(ts)} field={name} unique={len(np.unique(ts))} runs_of_equal={runs} "
              f"fields={list(fields)}")
        if i + 1 >= n:
            break
