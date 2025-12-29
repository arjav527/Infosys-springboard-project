"""
milestone1_ocr.py
Robust OCR extractor for medical lab reports (PDFs & images).

Outputs:
- extract_data_from_report(path) -> dict with parsed numeric fields and raw text for debugging.
  Example return:
    {
      "RestingBP": 130,
      "Cholesterol": 210,
      "raw_text": "full ocr text..."
    }

Notes:
- For PDF support install: pip install pdf2image
- On Windows, install Poppler and set POPPLER_PATH env var if needed.
- Requires Tesseract OCR installed and path set below.
"""

import os
from typing import Dict, Optional, List, Tuple
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import pytesseract
import re
import math

# Optional pdf -> images
try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except Exception:
    PDF_SUPPORT = False
    print("⚠️ pdf2image not installed. PDF scanning disabled. Run: pip install pdf2image if you need PDF support.")

# Allow overriding Poppler path via environment variable (Windows)
POPPLER_PATH = os.environ.get("POPPLER_PATH", r"C:\Program Files\poppler-24.02.0\Library\bin")

# Tesseract path - adjust if needed
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = os.environ.get("TESSERACT_CMD",
                                                           r"C:\Program Files\Tesseract-OCR\tesseract.exe")
else:
    pytesseract.pytesseract.tesseract_cmd = os.environ.get("TESSERACT_CMD", "/usr/bin/tesseract")


# -------------------------
# IMAGE PREPROCESSING
# -------------------------
def preprocess_image(img: Image.Image, upscale: int = 1200, threshold: int = 160, enhance: bool = True) -> Image.Image:
    """
    Convert to grayscale, upscale, sharpen, autocontrast and optionally apply binary threshold.
    Returns a PIL Image ready for OCR.
    """
    try:
        img = img.convert("L")
    except Exception:
        img = ImageOps.grayscale(img)

    # Upscale if small
    base_w = upscale
    if img.width < base_w:
        ratio = base_w / float(img.width)
        new_h = int(float(img.height) * ratio)
        img = img.resize((base_w, new_h), Image.LANCZOS)

    # Slight denoise/sharpen
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.filter(ImageFilter.SHARPEN)
    if enhance:
        img = ImageEnhance.Sharpness(img).enhance(1.2)
        img = ImageEnhance.Contrast(img).enhance(1.05)

    # Autocontrast and light threshold to B/W
    img = ImageOps.autocontrast(img)
    # threshold to B/W for clearer OCR
    img = img.point(lambda x: 0 if x < threshold else 255, mode="1")
    return img


def ocr_text_from_image(img: Image.Image, psm: int = 6, lang: str = "eng") -> str:
    """
    Run Tesseract OCR on PIL Image and return text.
    psm: page segmentation mode (6 = assume a single uniform block of text)
    """
    config = f"--psm {psm} -c preserve_interword_spaces=1"
    return pytesseract.image_to_string(img, config=config, lang=lang)


def pdf_to_images(pdf_path: str, max_pages: int = 3, dpi: int = 250) -> List[Image.Image]:
    """
    Convert a PDF to a list of PIL Images (first `max_pages` pages).
    Requires pdf2image and Poppler.
    """
    if not PDF_SUPPORT:
        raise RuntimeError("pdf2image not available. Install pdf2image to process PDFs.")
    if os.name == 'nt' and POPPLER_PATH:
        return convert_from_path(pdf_path, dpi=dpi, poppler_path=POPPLER_PATH)[:max_pages]
    else:
        return convert_from_path(pdf_path, dpi=dpi)[:max_pages]


# -------------------------
# TEXT CLEANING & UTIL
# -------------------------
def clean_text_for_regex(text: str) -> str:
    """
    Clean OCR text to reduce garbage but keep punctuation useful for patterns.
    """
    # Normalize common unicode, replace weird dashes, remove multiple spaces
    text = text.replace("\xa0", " ").replace("\u2013", "-").replace("\u2014", "-")
    # Keep digits, letters, common separators and mg/dl/mmhg symbols
    text = re.sub(r"[^0-9A-Za-z\s:./,()%\-µmMgGdDlL]", " ", text)
    # collapse spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_int_from_str(s: str) -> Optional[int]:
    """Parse integer from string with commas and decimals; return None if not parseable."""
    if s is None:
        return None
    s = s.strip().replace(",", "")
    m = re.search(r"\d+", s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        try:
            return int(float(s))
        except Exception:
            return None


def clamp_int(v: Optional[int], min_v: int, max_v: int) -> Optional[int]:
    if v is None:
        return None
    try:
        vi = int(v)
    except Exception:
        return None
    if vi < min_v or vi > max_v:
        return None
    return vi


# -------------------------
# PARSING HELPERS
# -------------------------
def find_nearest_number_around_keyword(text: str, keyword: str, window_chars: int = 40) -> Optional[int]:
    """
    Search for the keyword in text and return the first number found within a window around it.
    """
    for m in re.finditer(re.escape(keyword), text, flags=re.IGNORECASE):
        start = max(0, m.start() - window_chars)
        end = m.end() + window_chars
        snippet = text[start:end]
        nums = re.findall(r"\d{2,3}", snippet)
        if nums:
            return safe_int_from_str(nums[0])
    return None


def find_first_slash_bp(text: str) -> Optional[int]:
    """
    Looks for patterns like '120/80' and returns the systolic (first number).
    """
    m = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", text)
    if m:
        return safe_int_from_str(m.group(1))
    return None


# -------------------------
# MAIN EXTRACTOR
# -------------------------
def extract_data_from_report(path: str) -> Dict[str, Optional[int]]:
    """
    Extract RestingBP and Cholesterol from a PDF or image report.

    Returns:
      {
        "RestingBP": int or None,
        "Cholesterol": int or None,
        "raw_text": str
      }
    """
    parsed: Dict[str, Optional[int]] = {"RestingBP": None, "Cholesterol": None, "raw_text": ""}

    if not os.path.exists(path):
        print("❌ File not found:", path)
        return parsed

    ext = os.path.splitext(path)[1].lower()
    ocr_text = ""

    print(f"👁️ Scanning file: {path}")

    try:
        if ext == ".pdf":
            if not PDF_SUPPORT:
                raise RuntimeError("PDF support not available (pdf2image missing).")
            pages = pdf_to_images(path, max_pages=3)
            for i, page in enumerate(pages, start=1):
                print(f" → OCR page {i}/{len(pages)}")
                img = preprocess_image(page)
                ocr_text += "\n" + ocr_text_from_image(img, psm=6)
        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif"]:
            img = Image.open(path)
            img = preprocess_image(img)
            ocr_text = ocr_text_from_image(img, psm=6)
        else:
            print("⚠️ Unsupported extension:", ext)
            return parsed

        parsed["raw_text"] = ocr_text
        text = clean_text_for_regex(ocr_text).lower()

        # ---------------------------
        #  BLOOD PRESSURE (SYSTOLIC)
        # ---------------------------
        bp = None

        # 1) Look for labels followed by number (e.g., "Resting BP: 120 mmHg")
        bp_patterns = [
            r"(resting\s*blood\s*pressure|resting\s*bp|blood\s*pressure|bp|b\.p\.)\s*[:\-]?\s*(\d{2,3})",
            r"(systolic|sys)\s*[:\-]?\s*(\d{2,3})",
        ]
        for pat in bp_patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                # number may be in group 2 or group 1; pick last numeric group
                nums = re.findall(r"\d{2,3}", m.group(0))
                if nums:
                    bp = safe_int_from_str(nums[0])
                    break

        # 2) Try slash pattern 120/80
        if bp is None:
            bp = find_first_slash_bp(text)

        # 3) Try searching near keywords
        if bp is None:
            for kw in ["resting bp", "blood pressure", "bp", "systolic"]:
                found = find_nearest_number_around_keyword(text, kw)
                if found:
                    bp = found
                    break

        bp = clamp_int(bp, 70, 250)
        if bp:
            parsed["RestingBP"] = int(bp)
            print("   ✓ RestingBP:", bp)
        else:
            print("   ⚠ RestingBP not detected")

        # ---------------------------
        #  CHOLESTEROL
        # ---------------------------
        chol = None

        chol_patterns = [
            r"(serum\s*cholesterol|total\s*cholesterol|cholesterol|chol)\s*[:\-]?\s*(\d{2,3})",
            r"(ldl|hdl|triglyceride|tg)[^\d]{0,10}(\d{2,3})",
            r"(\d{2,3})\s*mg\s*/?\s*d[lL]?",
        ]

        for pat in chol_patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                nums = re.findall(r"\d{2,3}", m.group(0))
                if nums:
                    chol = safe_int_from_str(nums[0])
                    break

        # fallback: search near 'chol' keyword
        if chol is None:
            chol = find_nearest_number_around_keyword(text, "chol")

        # fallback: pick first 3-digit number that looks like cholesterol
        if chol is None:
            all_nums = re.findall(r"\d{2,3}", text)
            candidates = [int(n) for n in all_nums if 80 <= int(n) <= 450]
            if candidates:
                # pick the median-ish candidate to avoid low noise numbers
                try:
                    chol = int(sorted(candidates)[len(candidates)//2])
                except Exception:
                    chol = candidates[0] if candidates else None

        chol = clamp_int(chol, 80, 450)
        if chol:
            parsed["Cholesterol"] = int(chol)
            print("   ✓ Cholesterol:", chol)
        else:
            print("   ⚠ Cholesterol not detected")

        return parsed

    except Exception as e:
        print("❌ OCR Error:", e)
        return parsed


# -------------------------
# Small CLI helper for quick testing
# -------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="OCR extractor for heart reports (pdf/image).")
    p.add_argument("file", help="Path to PDF/Image file to scan")
    args = p.parse_args()
    out = extract_data_from_report(args.file)
    print("\n--- PARSED OUTPUT ---")
    print(out["raw_text"][:500], "...\n")
    print("Parsed values:", {k: v for k, v in out.items() if k != "raw_text"})
