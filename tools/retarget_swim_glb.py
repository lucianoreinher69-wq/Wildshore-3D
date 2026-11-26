#!/usr/bin/env python3
"""Retarget the bundled Quaternius swim clips onto Wildshore's existing camper rig."""
import base64
import lzma
import json
import math
import struct
import sys
from pathlib import Path

MAGIC = b"WSR2"
TARGET_NAMES = {
    "Hips", "Spine02", "Spine01", "Spine", "neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
}

def align4(data):
    return data + (b"\\0" * ((-len(data)) % 4))

def read_glb(path):
    data = Path(path).read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise SystemExit("Arquivo GLB inválido")
    version, total = struct.unpack_from("<II", data, 4)
    if version != 2 or total != len(data):
        raise SystemExit("Versão ou tamanho GLB inválido")
    offset = 12
    doc = None
    binary = None
    while offset < len(data):
        size, kind = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset:offset + size]
        offset += size
        if kind == 0x4E4F534A:
            doc = json.loads(chunk.decode("utf-8").rstrip(" \\t\\r\\n\\0"))
        elif kind == 0x004E4942:
            binary = chunk
    if doc is None or binary is None or not doc.get("buffers"):
        raise SystemExit("GLB sem JSON ou BIN")
    return doc, binary[:doc["buffers"][0]["byteLength"]]

def parse_payload(path):
    payload = lzma.decompress(base64.b64decode(Path(path).read_text(encoding="ascii").strip()))
    view = memoryview(payload)
    if bytes(view[:4]) != MAGIC:
        raise SystemExit("Dados de natação inválidos")
    offset = 4
    bone_count, clip_count = struct.unpack_from("<BB", view, offset)
    offset += 2
    bones = []
    for _ in range(bone_count):
        size = view[offset]
        offset += 1
        bones.append(bytes(view[offset:offset + size]).decode("ascii"))
        offset += size
    clips = []
    for _ in range(clip_count):
        size = view[offset]
        offset += 1
        name = bytes(view[offset:offset + size]).decode("ascii")
        offset += size
        duration, frame_count = struct.unpack_from("<fH", view, offset)
        offset += 6
        frames = []
        for _ in range(frame_count):
            time_ms = struct.unpack_from("<H", view, offset)[0]
            offset += 2
            rotations = []
            for _ in range(bone_count):
                q = struct.unpack_from("<4h", view, offset)
                offset += 8
                rotations.append([v / 32767.0 for v in q])
            hips = struct.unpack_from("<3h", view, offset)
            offset += 6
            frames.append((time_ms / 1000.0, rotations, [v / 100.0 for v in hips]))
        clips.append((name, duration, frames))
    if offset != len(payload):
        raise SystemExit("Dados de natação com tamanho inesperado")
    return bones, clips

def add_accessor(doc, binary, raw, component_type, accessor_type, count, minimum=None, maximum=None):
    binary = align4(binary)
    byte_offset = len(binary)
    binary += raw
    view_index = len(doc.setdefault("bufferViews", []))
    doc["bufferViews"].append({
        "buffer": 0,
        "byteOffset": byte_offset,
        "byteLength": len(raw),
    })
    accessor = {
        "bufferView": view_index,
        "componentType": component_type,
        "count": count,
        "type": accessor_type,
    }
    if minimum is not None:
        accessor["min"] = minimum
    if maximum is not None:
        accessor["max"] = maximum
    accessor_index = len(doc.setdefault("accessors", []))
    doc["accessors"].append(accessor)
    return binary, accessor_index

def write_glb(path, doc, binary):
    doc["buffers"][0]["byteLength"] = len(binary)
    json_bytes = json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    binary += b"\\0" * ((-len(binary)) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    out = (
        b"glTF" + struct.pack("<II", 2, total)
        + struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
        + struct.pack("<II", len(binary), 0x004E4942) + binary
    )
    Path(path).write_bytes(out)

def main():
    if len(sys.argv) != 3:
        raise SystemExit("Uso: retarget_swim_glb.py camper_rigged_animated.glb swim-retarget.bin.lzma.b64")
    model_path, data_path = sys.argv[1:]
    doc, binary = read_glb(model_path)
    bones, clips = parse_payload(data_path)
    if set(bones) != TARGET_NAMES or len(bones) != len(TARGET_NAMES):
        raise SystemExit("Mapa de ossos não corresponde ao personagem atual")
    node_by_name = {node.get("name"): i for i, node in enumerate(doc.get("nodes", []))}
    missing = sorted(TARGET_NAMES - set(node_by_name))
    if missing:
        raise SystemExit("O personagem não contém estes ossos: " + ", ".join(missing))
    existing = {a.get("name") for a in doc.get("animations", [])}
    for clip_name, duration, frames in clips:
        if clip_name in existing:
            raise SystemExit("A animação já existe no modelo: " + clip_name)
        if not frames or abs(frames[-1][0] - duration) > 0.002:
            raise SystemExit("Duração inválida na animação " + clip_name)
        times = [frame[0] for frame in frames]
        if any(times[i] >= times[i + 1] for i in range(len(times) - 1)):
            raise SystemExit("Tempos não crescentes na animação " + clip_name)
        raw_times = struct.pack("<" + "f" * len(times), *times)
        binary, time_accessor = add_accessor(
            doc, binary, raw_times, 5126, "SCALAR", len(times),
            [min(times)], [max(times)]
        )
        samplers = []
        channels = []
        for bone_index, bone_name in enumerate(bones):
            values = []
            for _, rotations, _ in frames:
                q = rotations[bone_index]
                length = math.sqrt(sum(v * v for v in q))
                if not math.isfinite(length) or length < 0.5:
                    raise SystemExit("Quaternion inválido: " + bone_name)
                values.extend(v / length for v in q)
            raw = struct.pack("<" + "f" * len(values), *values)
            binary, output_accessor = add_accessor(
                doc, binary, raw, 5126, "VEC4", len(times)
            )
            sampler_index = len(samplers)
            samplers.append({
                "input": time_accessor,
                "output": output_accessor,
                "interpolation": "LINEAR",
            })
            channels.append({
                "sampler": sampler_index,
                "target": {"node": node_by_name[bone_name], "path": "rotation"},
            })
        hips_values = []
        for _, _, hips in frames:
            hips_values.extend(hips)
        raw_hips = struct.pack("<" + "f" * len(hips_values), *hips_values)
        binary, hips_accessor = add_accessor(
            doc, binary, raw_hips, 5126, "VEC3", len(times)
        )
        sampler_index = len(samplers)
        samplers.append({
            "input": time_accessor,
            "output": hips_accessor,
            "interpolation": "LINEAR",
        })
        channels.append({
            "sampler": sampler_index,
            "target": {"node": node_by_name["Hips"], "path": "translation"},
        })
        doc.setdefault("animations", []).append({
            "name": clip_name,
            "samplers": samplers,
            "channels": channels,
        })
    write_glb(model_path, doc, binary)
    check, _ = read_glb(model_path)
    names = {a.get("name") for a in check.get("animations", [])}
    required = {"Walk", "Run", "Swim_Fwd_Loop", "Swim_Idle_Loop"}
    if not required.issubset(names):
        raise SystemExit("O GLB final não contém todas as animações necessárias")
    print("GLB retargetado:", Path(model_path).stat().st_size, "bytes")
    print("Animações:", ", ".join(sorted(names)))

if __name__ == "__main__":
    main()
