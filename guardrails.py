import cv2
import numpy as np

def check_resolution(image, min_width=224, min_height=224):
    """Checks if the image meets the minimum resolution requirement."""
    height, width = image.shape[:2]
    if width < min_width or height < min_height:
        return False, f"Resolution too low: {width}x{height}. Minimum is {min_width}x{min_height}."
    return True, f"Resolution OK: {width}x{height}"

def check_blur(image, threshold=15.0):
    """
    Detects blur using the variance of the Laplacian method.
    A lower variance means the image is blurrier.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if variance < threshold:
        return False, f"Image is blurry (Score: {variance:.2f} < {threshold})"
    return True, f"Image sharpness OK (Score: {variance:.2f})"

def check_brightness(image, min_brightness=40, max_brightness=220):
    """
    Checks if the image is too dark or washed out (overexposed).
    """
    # Convert to HSV color space to easily extract brightness (Value channel)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])
    
    if brightness < min_brightness:
        return False, f"Image is too dark (Brightness: {brightness:.2f})"
    elif brightness > max_brightness:
        return False, f"Image is overexposed/washed out (Brightness: {brightness:.2f})"
    
    return True, f"Image brightness OK (Brightness: {brightness:.2f})"

def check_is_fundus(image):
    """
    Basic heuristic to check if it's a fundus image.
    Fundus images typically have a large dark circular border.
    We check the corners of the image; if they are mostly black, it's likely a fundus image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Check 4 corners (e.g., 10x10 squares)
    corners = [
        gray[0:10, 0:10],
        gray[0:10, width-10:width],
        gray[height-10:height, 0:10],
        gray[height-10:height, width-10:width]
    ]
    
    corner_brightness = np.mean([np.mean(c) for c in corners])
    
    # If the corners are extremely dark (close to 0), it has the circular fundus mask
    if corner_brightness > 50:
        return False, f"Missing fundus circular structure (Corners too bright: {corner_brightness:.2f})"
    return True, f"Fundus structure detected (Dark corners: {corner_brightness:.2f})"

def run_all_guardrails(image_path):
    """Runs all checks on a given image path."""
    print(f"--- Running Guardrails for {image_path} ---")
    
    # 1. Image Load Check
    image = cv2.imread(image_path)
    if image is None:
        return {"status": "FAILED", "reason": "Could not read the image file. Invalid format or corrupted."}
    
    passed_all = True
    report = []
    
    # Check Resolution
    res_passed, res_msg = check_resolution(image)
    report.append(res_msg)
    if not res_passed: passed_all = False
        
    # Check Blur
    blur_passed, blur_msg = check_blur(image)
    report.append(blur_msg)
    if not blur_passed: passed_all = False
        
    # Check Brightness
    bright_passed, bright_msg = check_brightness(image)
    report.append(bright_msg)
    if not bright_passed: passed_all = False
        
    # Check Fundus Structure
    fundus_passed, fundus_msg = check_is_fundus(image)
    report.append(fundus_msg)
    if not fundus_passed: passed_all = False
        
    final_status = "PASSED" if passed_all else "FAILED"
    
    print(f"Final Status: {final_status}")
    for msg in report:
        print(f" - {msg}")
        
    return {"status": final_status, "details": report}

if __name__ == "__main__":
    # Test on a single image from your extracted dataset
    test_image_path = r"C:\DR_CP\DE_MESSIDOR_EX\augmented_resized_V2\test\0\0212dd31f623-600.jpg" 
    
    print(f"Testing guardrails on: {test_image_path}")
    result = run_all_guardrails(test_image_path)
    print("\n--- Final Guardrail Result ---")
    print(result)
