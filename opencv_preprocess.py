import cv2
import numpy as np

def enhance_for_ocr(image_path_or_bytes):
    """
    Enhance an image for OCR using OpenCV. Accepts a file path or bytes.
    Attempts to detect and crop to the card area before further processing.
    Returns the processed (deskewed, binarized) image as a numpy array (uint8).
    """
    # Accept either file path or bytes
    if isinstance(image_path_or_bytes, str):
        img = cv2.imread(image_path_or_bytes)
    else:
        img_array = np.frombuffer(image_path_or_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not load image for OpenCV processing.")
    orig = img.copy()
    h, w = img.shape[:2]
    target_w = 1024
    scale = target_w / float(w)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # --- Card Area Detection ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_contour = None
    max_area = 0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        area = cv2.contourArea(cnt)
        if len(approx) == 4 and area > max_area and area > 0.2 * (img.shape[0] * img.shape[1]):
            card_contour = approx
            max_area = area
    if card_contour is not None:
        pts = card_contour.reshape(4, 2)
        # Order the points: top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        img = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        # --- Further robust enhancement ---
        # 1. Adaptive gamma correction
        def adjust_gamma(image, gamma=1.0):
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, table)
        # Estimate gamma from mean brightness
        mean_brightness = np.mean(img)
        gamma = 1.0
        if mean_brightness < 100:
            gamma = 1.5
        elif mean_brightness > 180:
            gamma = 0.7
        img = adjust_gamma(img, gamma)
        # 2. Shadow removal (illumination normalization)
        rgb_planes = cv2.split(img)
        result_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((15,15), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            result_planes.append(norm_img)
        img = cv2.merge(result_planes)
        # 3. Upscale
        img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
        # 4. Sharpening (twice)
        sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, sharpening_kernel)
        img = cv2.filter2D(img, -1, sharpening_kernel)
        # 5. Contrast/brightness boost
        img = cv2.convertScaleAbs(img, alpha=1.3, beta=25)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # --- Continue with enhancement ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=30, templateWindowSize=7, searchWindowSize=21)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    closed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
    th = cv2.adaptiveThreshold(
        closed, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=25,
        C=10
    )
    if np.mean(th) < 127:
        th = cv2.bitwise_not(th)
    coords = np.column_stack(np.where(th > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        M = cv2.getRotationMatrix2D((th.shape[1]/2, th.shape[0]/2), angle, 1)
        deskewed = cv2.warpAffine(th, M, (th.shape[1], th.shape[0]),
                                  flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else:
        deskewed = th
    return deskewed
