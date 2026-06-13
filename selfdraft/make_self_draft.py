#!/usr/bin/env python3
"""
make_self_draft.py - derive a cheap DRAFT gguf from a target model's OWN layers.

Builds a standalone GGUF containing a subset of the target's transformer layers
(+ its embedding / output_norm / tied head), so llama.cpp's lossless `draft-simple`
path can use it as a self-speculative draft. No MTP head, no download - works on any
model. Quantized tensor bytes are copied verbatim (never requantized).

Stage 1 (first-K):   --keep K           -> keep layers 0..K-1
Stage 2 (salience):  --layers i,j,k,... -> keep+reorder arbitrary layers (remapped 0..M-1)

Run with the repo's gguf-py on the path:
  PYTHONPATH=llama.cpp/gguf-py python3 selfdraft/make_self_draft.py --in T.gguf --out D.gguf --keep 24
"""
import argparse, re, sys, os
import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType

def subset_and_remap(src_path, dst_path, keep_layers):
    K = len(keep_layers)
    old2new = {old: new for new, old in enumerate(keep_layers)}
    blk_re = re.compile(r'^blk\.(\d+)\.(.*)$')

    reader = GGUFReader(src_path, 'r')
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    writer = GGUFWriter(dst_path, arch=arch, endianess=reader.endianess)

    align = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if align is not None:
        writer.data_alignment = align.contents()

    block_count_key = f'{arch}.block_count'
    old_bc = reader.get_field(block_count_key)
    old_n = old_bc.contents() if old_bc is not None else None
    if old_n is not None:
        for i in keep_layers:
            if i < 0 or i >= old_n:
                sys.exit(f"ERROR: layer index {i} out of range (model has {old_n} layers)")

    # metadata: verbatim, except block_count -> K and per-layer arrays (len==old_n) resliced
    resliced = []
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
            continue
        vt = field.types[0]
        sub = field.types[-1] if vt == GGUFValueType.ARRAY else None
        if field.name == block_count_key:
            writer.add_key_value(field.name, K, vt)
            continue
        val = field.contents()
        if vt == GGUFValueType.ARRAY and isinstance(val, list) and old_n is not None and len(val) == old_n:
            val = [val[old] for old in keep_layers]
            resliced.append(field.name)
        writer.add_key_value(field.name, val, vt, sub_type=sub)
    if old_n is None:
        writer.add_uint32(block_count_key, K)

    # tensors: non-layer verbatim; blk.{i}.* kept+remapped to 0..K-1
    selected, dropped = [], 0
    for t in reader.tensors:
        m = blk_re.match(t.name)
        if m is None:
            selected.append((t.name, t)); continue
        old_idx = int(m.group(1))
        if old_idx not in old2new:
            dropped += 1; continue
        selected.append((f'blk.{old2new[old_idx]}.{m.group(2)}', t))

    for name, t in selected:
        writer.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for name, t in selected:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    writer.close()

    print(f"[make_self_draft] arch={arch} kept {K}/{old_n} layers -> {os.path.basename(dst_path)}")
    print(f"  layers: {keep_layers}")
    print(f"  resliced per-layer arrays: {resliced}")
    print(f"  tensors written: {len(selected)} (dropped {dropped} layer tensors)")
    print(f"  size: {os.path.getsize(dst_path)/1e9:.2f} GB (target {os.path.getsize(src_path)/1e9:.2f} GB)")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--keep", type=int, help="keep first K layers (0..K-1)")
    g.add_argument("--layers", help="comma list of source layer indices to keep+reorder, e.g. 0,1,2,5,9")
    a = ap.parse_args()
    keep = list(range(a.keep)) if a.keep is not None else [int(x) for x in a.layers.split(",")]
    subset_and_remap(a.src, a.dst, keep)

if __name__ == "__main__":
    main()
