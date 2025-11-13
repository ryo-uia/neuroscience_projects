from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageOps
import argparse
import sys

def pick_file() -> Path | None:
    """Try GUI picker; fall back to console input."""
    try:
        from tkinter import Tk, filedialog
        Tk().withdraw()
        sel = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp;*.webp"),
                       ("All files", "*.*")]
        )
        if sel:
            return Path(sel)
    except Exception:
        pass
    p = input("Enter image path: ").strip().strip('"')
    return Path(p) if p else None

def mirror_one(inp: Path, out: Path | None, autorotate: bool, overwrite: bool, suffix: str) -> Path:
    if not inp.exists():
        raise FileNotFoundError(inp)
    dst = out or inp.with_name(f"{inp.stem}{suffix}{inp.suffix}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {dst} (use -f to overwrite)")

    with Image.open(inp) as im:
        if autorotate:
            im = ImageOps.exif_transpose(im)   # normalize orientation first
        mirrored = ImageOps.mirror(im)

        # Prepare optional EXIF only for safe formats
        save_kwargs = {}
        try:
            exif = im.getexif()
        except Exception:
            exif = None

        # Drop Orientation tag if present (274) to avoid viewer confusion
        if exif and 274 in exif:
            try:
                del exif[274]
            except Exception:
                pass

        ext = dst.suffix.lower()
        if exif and ext in {".jpg", ".jpeg", ".webp"}:
            # Only attach EXIF bytes for formats that handle it well
            try:
                save_kwargs["exif"] = exif.tobytes()
            except Exception:
                pass

        try:
            mirrored.save(dst, **save_kwargs)
        except Exception:
            # Fallback: save without EXIF if the format/encoder complains
            mirrored.save(dst)

    return dst

def main():
    ap = argparse.ArgumentParser(description="Create a mirror (left↔right) copy of one image.")
    ap.add_argument("input", nargs="?", help="Path to input image (omit to choose via dialog/prompt)")
    ap.add_argument("-o", "--output", help="Output file path (default: <name>_mirrored<ext> next to input)")
    ap.add_argument("-f", "--force", dest="overwrite", action="store_true", help="Overwrite output if it exists")
    ap.add_argument("--no-autorotate", dest="autorotate", action="store_false",
                    help="Disable EXIF-based autorotation before mirroring")
    ap.add_argument("--suffix", default="_mirrored", help="Suffix for auto-named output (default: _mirrored)")
    args = ap.parse_args()

    inp = Path(args.input) if args.input else pick_file()
    if not inp:
        print("No file selected. Exiting.")
        sys.exit(1)

    out = Path(args.output) if args.output else None
    dst = mirror_one(inp, out, autorotate=args.autorotate, overwrite=args.overwrite, suffix=args.suffix)
    print(f"Saved: {dst}")

if __name__ == "__main__":
    main()