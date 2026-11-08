# save_lean as of git 10a26c4 (before w78's row-block save), for test_w78.py's byte compare
def save_lean(path, tot, metadata):
    """safetensors file of tot's "H", "C", "N" as f32, written one tensor at a time: host RAM holds one k x k f32
    (1.2 GB at 17408), not save_file's three plus their serialized copies (~7 GB). Same tensors and metadata
    (safe_open reads it as save_file's; header padded with spaces to 8 bytes as the format allows)."""
    hdr, off = {}, 0
    for s in "HCN":
        k = tot[s].shape; nb = 4 * k[0] * k[1]
        hdr[s] = {"dtype": "F32", "shape": list(k), "data_offsets": [off, off + nb]}; off += nb
    hdr["__metadata__"] = metadata
    hb = json.dumps(hdr, separators=(",", ":")).encode(); hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as fh:
        fh.write(len(hb).to_bytes(8, "little")); fh.write(hb)
        for s in "HCN":
            t = tot[s].float().cpu().contiguous(); t.numpy().tofile(fh); del t
    assert os.path.getsize(path) == 8 + len(hb) + off, path
