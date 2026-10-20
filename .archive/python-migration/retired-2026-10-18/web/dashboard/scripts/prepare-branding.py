#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Prepare dashboard artwork using the managed image's locked native libraries."""
import argparse
import ctypes as c
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DASHBOARD = ROOT / "web/dashboard"
SOURCES = ROOT / "web/branding"
DESTINATION = DASHBOARD / "assets/branding"
BUDGET = 512 * 1024
QUALITY = 82


def bind(library, name, result, *arguments):
    function = getattr(library, name)
    function.restype = result
    function.argtypes = list(arguments)
    return function


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if sys.platform != "linux" or not Path("/.dockerenv").is_file() or ROOT != Path("/mityane"):
        raise SystemExit("Use container.ps1 and the managed mityane-geometrics container.")
    selection = json.loads((SOURCES / "dashboard-assignments.json").read_text(encoding="utf-8-sig"))
    manifest = json.loads((DASHBOARD / "manifest.json").read_text(encoding="utf-8-sig"))
    ids = [hero["id"] for hero in manifest["heroes"]]
    if set(selection) != {"global", "heroes"} or set(selection["heroes"]) != set(ids):
        raise ValueError("Assign artwork to the global brand and every manifest Hero, with no extra IDs.")
    selected = [("global", selection["global"], 384)] + [
        (hero_id, selection["heroes"][hero_id], 256) for hero_id in ids
    ]
    if len({source for _, source, _ in selected}) != len(selected):
        raise ValueError("Each Hero and the global brand need distinct source artwork.")

    pixbuf = c.CDLL("libgdk_pixbuf-2.0.so.0")
    webp = c.CDLL("libwebp.so.7")
    objects = c.CDLL("libgobject-2.0.so.0")
    pixels_type = c.POINTER(c.c_uint8)
    load = bind(pixbuf, "gdk_pixbuf_new_from_file", c.c_void_p, c.c_char_p, c.POINTER(c.c_void_p))
    scale = bind(pixbuf, "gdk_pixbuf_scale_simple", c.c_void_p, c.c_void_p, c.c_int, c.c_int, c.c_int)
    pixels = bind(pixbuf, "gdk_pixbuf_get_pixels", pixels_type, c.c_void_p)
    stride = bind(pixbuf, "gdk_pixbuf_get_rowstride", c.c_int, c.c_void_p)
    channels = bind(pixbuf, "gdk_pixbuf_get_n_channels", c.c_int, c.c_void_p)
    width = bind(pixbuf, "gdk_pixbuf_get_width", c.c_int, c.c_void_p)
    height = bind(pixbuf, "gdk_pixbuf_get_height", c.c_int, c.c_void_p)
    unref = bind(objects, "g_object_unref", None, c.c_void_p)
    encode = bind(webp, "WebPEncodeRGBA", c.c_size_t, pixels_type, c.c_int, c.c_int,
                  c.c_int, c.c_float, c.POINTER(pixels_type))
    free = bind(webp, "WebPFree", None, c.c_void_p)
    info = bind(webp, "WebPGetInfo", c.c_int, c.c_void_p, c.c_size_t,
                c.POINTER(c.c_int), c.POINTER(c.c_int))
    version = bind(webp, "WebPGetEncoderVersion", c.c_int)()
    files = {}
    records = []
    for identity, relative, size in selected:
        source = (SOURCES / relative).resolve()
        if not source.is_relative_to(SOURCES) or source.suffix != ".png":
            raise ValueError("Source artwork must be a PNG inside web/branding/.")
        original = load(str(source).encode(), None)
        if not original:
            raise ValueError(f"Cannot decode source artwork: {relative}")
        scaled = None
        try:
            if channels(original) != 4 or width(original) != height(original):
                raise ValueError(f"Expected square RGBA artwork: {relative}")
            scaled = scale(original, size, size, 2)  # GDK_INTERP_BILINEAR
            if not scaled:
                raise ValueError(f"Cannot resize artwork: {relative}")
            output = pixels_type()
            length = encode(pixels(scaled), size, size, stride(scaled), QUALITY, c.byref(output))
            if not length:
                raise ValueError(f"Cannot encode artwork: {relative}")
            try:
                data = c.string_at(output, length)
            finally:
                free(output)
        finally:
            if scaled:
                unref(scaled)
            unref(original)
        decoded_width, decoded_height = c.c_int(), c.c_int()
        buffer = c.create_string_buffer(data)
        if not info(buffer, len(data), c.byref(decoded_width), c.byref(decoded_height)) or (
            decoded_width.value, decoded_height.value
        ) != (size, size):
            raise ValueError(f"Encoded artwork failed verification: {relative}")
        filename = f"{identity}.webp"
        files[filename] = data
        records.append({"id": identity, "source": str(source.relative_to(ROOT)),
                        "sourceSHA256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "file": filename, "sha256": hashlib.sha256(data).hexdigest(),
                        "width": size, "height": size, "bytes": len(data)})
    total = sum(map(len, files.values()))
    if total > BUDGET:
        raise ValueError(f"Prepared artwork exceeds the unchanged 512 KiB budget: {total} bytes.")
    config = {"mark": "global.webp", "portrait": "global.webp", "defaultHero": "global.webp",
              "heroes": {hero_id: f"{hero_id}.webp" for hero_id in ids}}
    provenance = {"source": "web/branding/dashboard-assignments.json", "format": "WebP RGBA",
                  "quality": QUALITY, "resize": "GdkPixbuf bilinear",
                  "encoder": f"libwebp {version >> 16}.{(version >> 8) & 255}.{version & 255}",
                  "totalBytes": total, "budgetBytes": BUDGET, "assets": records}
    for filename, value in [("branding.json", config), ("provenance.json", provenance)]:
        files[filename] = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    for filename, data in files.items():
        target = DESTINATION / filename
        if target.is_file() and target.read_bytes() == data:
            continue
        if arguments.check:
            raise ValueError(f"Prepared branding is out of date: {filename}. Run refresh.ps1.")
        target.write_bytes(data)
    verb = "Verified" if arguments.check else "Prepared"
    print(f"{verb} global artwork and {len(ids)} unique Hero images: {total} / {BUDGET} bytes.")


if __name__ == "__main__":
    main()
