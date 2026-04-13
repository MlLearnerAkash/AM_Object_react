import h5py
import numpy as np
import pickle
import io

def save_episode_to_hdf5(hdf5_path: str, ep_data: dict) -> None:
    """
    Append one episode into an HDF5 file.

    HDF5 layout
    -----------
    /<episode_id>/
        instruction          (scalar string dataset)
        graph                (raw bytes dataset — pickled networkx graph)
        frames/
            <000>/
                rgb          (H x W x 3  uint8)
                frame_idx    (scalar int)
            <001>/
                ...
    """
    episode_id = str(ep_data["episode_id"])

    with h5py.File(hdf5_path, "a") as hf:          # "a" = append / create
        if episode_id in hf:
            del hf[episode_id]                      # overwrite if re-running

        ep_grp = hf.create_group(episode_id)

        # ── instruction ───────────────────────────────────────────────────
        ep_grp.create_dataset(
            "instruction",
            data=ep_data["instruction"],            # h5py stores str natively
        )

        # ── episode graph (pickle → bytes → HDF5 opaque blob) ────────────
        buf = io.BytesIO()
        pickle.dump(ep_data["graph"], buf)
        graph_bytes = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        ep_grp.create_dataset(
            "graph",
            data=graph_bytes,
            compression="gzip",
            compression_opts=4,
        )

        # ── per-frame RGB images ──────────────────────────────────────────
        frames_grp = ep_grp.create_group("frames")

        for fd in ep_data["frame_data"]:
            frame_key = f"{fd['frame_idx']:03d}"
            frame_grp = frames_grp.create_group(frame_key)

            rgb = np.asarray(fd["rgb"], dtype=np.uint8)   # ensure ndarray
            frame_grp.create_dataset(
                "rgb",
                data=rgb,
                compression="gzip",
                compression_opts=4,
            )
            frame_grp.create_dataset(
                "frame_idx",
                data=fd["frame_idx"],
            )

def load_episode_from_hdf5(hdf5_path: str, episode_id: str) -> dict:
    """
    Read one episode back from the HDF5 file.

    Returns a dict with keys:
        episode_id  : str
        instruction : str
        graph       : original Python object (unpickled)
        frame_data  : list of {"frame_idx": int, "rgb": np.ndarray}
    """
    episode_id = str(episode_id)

    with h5py.File(hdf5_path, "r") as hf:
        if episode_id not in hf:
            raise KeyError(f"Episode '{episode_id}' not found in {hdf5_path}")

        ep_grp = hf[episode_id]

        instruction = ep_grp["instruction"][()].decode("utf-8") \
            if isinstance(ep_grp["instruction"][()], bytes) \
            else str(ep_grp["instruction"][()])

        graph_bytes = ep_grp["graph"][()].tobytes()
        graph = pickle.loads(graph_bytes)

        frame_data = []
        frames_grp = ep_grp["frames"]
        for frame_key in sorted(frames_grp.keys()):
            fg = frames_grp[frame_key]
            frame_data.append({
                "frame_idx": int(fg["frame_idx"][()]),
                "rgb":       fg["rgb"][()],             # np.ndarray uint8
            })

    return {
        "episode_id":  episode_id,
        "instruction": instruction,
        "graph":       graph,
        "frame_data":  frame_data,
    }