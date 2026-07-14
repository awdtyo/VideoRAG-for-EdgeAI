"""
On-screen text extraction via Tesseract. Kept intentionally sparse (only
every Nth sampled frame, see config.OCR_EVERY_N_FRAMES) since Tesseract is
CPU-heavy relative to everything else in this pipeline.
"""
import cv2
import pytesseract


def extract_text(frame_bgr, min_len=3):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # light denoise + threshold helps Tesseract on video frames (motion blur, compression)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    text = pytesseract.image_to_string(thresh).strip()
    if len(text) < min_len:
        return ""
    return " ".join(text.split())
