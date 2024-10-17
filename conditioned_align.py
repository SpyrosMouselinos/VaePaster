import cv2
import numpy as np
import pytesseract
from paddleocr import PaddleOCR
import os
from PIL import Image
import random
import yaml
import argparse
import glob
import gc

class LabelTransformer:
    def __init__(self, ocr_engine='paddle', config_path='settings.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.sift = cv2.SIFT_create(self.config['sift']['n_features'])
        self.matcher = cv2.BFMatcher(
            cv2.NORM_L2 if self.config['matcher']['norm'] == 'L2' else cv2.NORM_HAMMING,
            crossCheck=self.config['matcher']['cross_check']
        )
        self.ocr_engine = ocr_engine
        if ocr_engine == 'paddle':
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

    def detect_text_regions(self, img):
        if self.ocr_engine == 'paddle':
            return self._detect_text_regions_paddle(img)
        else:
            return self._detect_text_regions_pytesseract(img)

    def _detect_text_regions_paddle(self, img):
        print("Starting text detection with PaddleOCR")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.ocr.ocr(img_rgb, cls=True)[0]
        boxes = []
        for item in result:
            box = item[0]
            text = item[1][0]
            conf = item[1][1]
            if conf >= self.config['text_detection']['confidence_threshold']:
                x = min(box[0][0], box[3][0])
                y = min(box[0][1], box[1][1])
                w = max(box[1][0], box[2][0]) - x
                h = max(box[2][1], box[3][1]) - y
                boxes.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h), 'conf': conf, 'text': text})
        return boxes

    def _detect_text_regions_pytesseract(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        custom_config = r'--oem 3 --psm 11'
        data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT, config=custom_config)

        boxes = []
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            conf = int(data['conf'][i])
            if conf >= 70:  # Confidence threshold
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                text = data['text'][i]
                boxes.append({'x': x, 'y': y, 'w': w, 'h': h, 'conf': conf, 'text': text})
        return boxes

    def transform_label(self, ref_image, transformed_image=None, save_warped_labels=None, overlay_labels=None):
        save_warped_labels = self.config['display']['save_warped_labels'] if save_warped_labels is None else save_warped_labels
        overlay_labels = self.config['display']['overlay_labels'] if overlay_labels is None else overlay_labels

        print(f"transform_label called with:")
        print(f"  ref_image shape: {ref_image.shape}")
        
        if transformed_image is None:
            print("Error: transformed_image is None in conditioned_align.py")
            return None, None
        else:
            print(f"  transformed_image shape: {transformed_image.shape}")
        
        print(f"  save_warped_labels: {save_warped_labels}")
        print(f"  overlay_labels: {overlay_labels}")

        # Detect text regions in ref_image
        text_regions = self.detect_text_regions(ref_image)
        if text_regions is None:
            print("Could not detect labels via OCR.")
            return None, None

        # Find the homography matrix from transformed_image to ref_image
        keypoints1, descriptors1 = self.sift.detectAndCompute(ref_image, None)
        keypoints2, descriptors2 = self.sift.detectAndCompute(transformed_image, None)
        
        matches = self.matcher.match(descriptors2, descriptors1)
        matches = sorted(matches, key=lambda x: x.distance)

        src_pts = np.float32([keypoints2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # This M transforms points from transformed_image to ref_image
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Invert M to get the transformation from ref_image to transformed_image
        M_inv = np.linalg.inv(M)

        warped_labels = []
        result_img = transformed_image.copy() if overlay_labels else None

        for idx, region in enumerate(text_regions):
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            # Create a box for the text region
            box = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]]).reshape(-1, 1, 2)
            
            # Transform the box using the inverse homography
            transformed_box = cv2.perspectiveTransform(box, M_inv)
            
            # Get the bounding rectangle of the transformed box
            rect = cv2.boundingRect(transformed_box)
            tx, ty, tw, th = rect
            
            # Extract the warped label from the transformed image
            warped_label = transformed_image[ty:ty+th, tx:tx+tw]
            warped_labels.append(warped_label)

            if overlay_labels:
                # Draw the transformed box on the result image
                cv2.polylines(result_img, [np.int32(transformed_box)], True, (0, 255, 0), 2)
                cv2.putText(result_img, region['text'], (tx, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return warped_labels, result_img

    def auto_crop_and_correct(self, ref_image, transformed_image):
        # Convert images to grayscale
        ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
        transformed_gray = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)

        # Find keypoints and descriptors
        keypoints1, descriptors1 = self.sift.detectAndCompute(ref_gray, None)
        keypoints2, descriptors2 = self.sift.detectAndCompute(transformed_gray, None)

        # Match descriptors
        matches = self.matcher.match(descriptors1, descriptors2)

        # Sort them in ascending order of distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Choose the top N matches
        N = 200
        good_matches = matches[:N]

        # Extract the matched keypoints
        src_pts = np.float32([keypoints1[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)

        # Find the homography matrix
        M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        # Use the homography matrix to warp the transformed image
        corrected_image = cv2.warpPerspective(transformed_image, M, (ref_image.shape[1], ref_image.shape[0]))

        return ref_image, corrected_image

    def process_images(self, ref_image, transformed_image):
        # ACC (Automatic Crop and Correct)
        ref_image, transformed_image = self.auto_crop_and_correct(ref_image, transformed_image)

        # Calculate the pixel size of the original reference and transformed images
        ref_height, ref_width, _ = ref_image.shape
        transformed_height, transformed_width, _ = transformed_image.shape

        print(f"Reference Image Size (pixels): {ref_width} x {ref_height}")
        print(f"Transformed Image Size (pixels): {transformed_width} x {transformed_height}")

        # Convert images to grayscale
        ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
        transformed_gray = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)

        # Detect SIFT features and compute descriptors
        keypoints1, descriptors1 = self.sift.detectAndCompute(ref_gray, None)
        keypoints2, descriptors2 = self.sift.detectAndCompute(transformed_gray, None)

        # Match features
        matches = list(self.matcher.match(descriptors1, descriptors2, None))

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not-so-good matches
        numGoodMatches = int(len(matches) * 0.5)
        matches = matches[:numGoodMatches]

        # Calculate and display the percentage and number of feature matches
        total_matches = len(matches)
        total_keypoints1 = len(keypoints1)
        total_keypoints2 = len(keypoints2)
        match_percentage = (total_matches / min(total_keypoints1, total_keypoints2)) * 100

        print(f"Number of feature matches: {total_matches}")
        print(f"Percentage of feature matches: {match_percentage:.2f}%")

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

        # Use homography to warp image
        height, width, channels = ref_image.shape
        transformed_image_aligned = cv2.warpPerspective(transformed_image, h, (width, height))

        # Calculate the pixel size of the aligned transformed image
        aligned_height, aligned_width, _ = transformed_image_aligned.shape
        print(f"Aligned Transformed Image Size (pixels): {aligned_width} x {aligned_height}")

        return ref_image, transformed_image_aligned

def resize_image(image, max_size=1500):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def apply_mosaic_blur(image, box, block_size=20, blur_percentage=0.6):
    x, y, w, h = cv2.boundingRect(box)
    center_w = int(w * blur_percentage)
    center_h = int(h * blur_percentage)
    start_x = x + (w - center_w) // 2
    start_y = y + (h - center_h) // 2
    region = image[start_y:start_y+center_h, start_x:start_x+center_w]
    h, w = region.shape[:2]
    mosaic = cv2.resize(region, (block_size, block_size), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(mosaic, (w, h), interpolation=cv2.INTER_NEAREST)
    result = image.copy()
    result[start_y:start_y+center_h, start_x:start_x+center_w] = mosaic
    return result

def paste_original_text(ref_image, transformed_image, text_regions, M_inv):
    result = transformed_image.copy()
    
    for region in text_regions:
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        
        # Create a box for the text region
        box = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]]).reshape(-1, 1, 2)
        
        # Transform the box using the inverse homography
        transformed_box = cv2.perspectiveTransform(box, M_inv)
        
        # Get the bounding rectangle of the transformed box
        rect = cv2.boundingRect(np.int32(transformed_box))
        tx, ty, tw, th = rect
        
        # Extract the text region from the original image
        original_text = ref_image[y:y+h, x:x+w]
        
        # Resize the extracted text to fit the transformed box
        resized_text = cv2.resize(original_text, (tw, th), interpolation=cv2.INTER_LINEAR)
        
        # Create a mask for smooth blending
        mask = np.zeros((th, tw), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(transformed_box) - np.int32([tx, ty]), 1)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        
        # Blend the resized text onto the result image
        for c in range(3):  # for each color channel
            result[ty:ty+th, tx:tx+tw, c] = (
                result[ty:ty+th, tx:tx+tw, c] * (1 - mask) +
                resized_text[:, :, c] * mask
            )
    
    return result

def process_image_pair(label_transformer, original_path, generated_path, output_dir):
    try:
        ref_image = cv2.imread(original_path)
        transformed_image = cv2.imread(generated_path)
        
        if ref_image is None or transformed_image is None:
            print(f"Error: Unable to load images from {original_path} or {generated_path}")
            return

        print(f"Processing image pair:")
        print(f"Original: {original_path}")
        print(f"Generated: {generated_path}")
        print(f"Original Image shape: {ref_image.shape}")
        print(f"Generated Image shape: {transformed_image.shape}")

        ref_image = resize_image(ref_image)
        transformed_image = resize_image(transformed_image)
        print(f"Resized Original Image shape: {ref_image.shape}")
        print(f"Resized Generated Image shape: {transformed_image.shape}")

        # Save transformed image without annotations
        transformed_image_path = os.path.join(output_dir, 'generated_image.png')
        cv2.imwrite(transformed_image_path, transformed_image)
        print(f"Generated image saved as '{transformed_image_path}'")

        # Detect text regions in ref_image
        text_regions = label_transformer.detect_text_regions(ref_image)

        # Create a copy of transformed_image for applying mosaic blur to text regions
        transformed_blurred = transformed_image.copy()

        # Find the homography matrix from transformed_image to ref_image
        keypoints1, descriptors1 = label_transformer.sift.detectAndCompute(ref_image, None)
        keypoints2, descriptors2 = label_transformer.sift.detectAndCompute(transformed_image, None)
        
        matches = label_transformer.matcher.match(descriptors2, descriptors1)
        matches = sorted(matches, key=lambda x: x.distance)

        src_pts = np.float32([keypoints2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # This M transforms points from transformed_image to ref_image
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Invert M to get the transformation from ref_image to transformed_image
        M_inv = np.linalg.inv(M)

        # Apply mosaic blur to text regions in transformed_blurred
        for region in text_regions:
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            # Create a box for the text region
            box = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]]).reshape(-1, 1, 2)
            
            # Transform the box using the inverse homography
            transformed_box = cv2.perspectiveTransform(box, M_inv)
            
            # Convert to integer coordinates
            transformed_box = np.int32(transformed_box)
            
            # Apply mosaic blur to the central 60% of the region
            transformed_blurred = apply_mosaic_blur(transformed_blurred, transformed_box, block_size=10, blur_percentage=0.6)

        # Paste original text onto the transformed image
        transformed_with_original_text = paste_original_text(ref_image, transformed_image, text_regions, M_inv)

        # Save the transformed image with original text pasted
        transformed_original_text_path = os.path.join(output_dir, 'generated_with_original_text.png')
        cv2.imwrite(transformed_original_text_path, transformed_with_original_text)
        print(f"Generated image with original text pasted saved as '{transformed_original_text_path}'")

        warped_labels, result_img = label_transformer.transform_label(
            ref_image,
            transformed_image,
            save_warped_labels=True,
            overlay_labels=True
        )

        if result_img is not None:
            result_filename = os.path.join(output_dir, 'result_with_annotations.png')
            cv2.imwrite(result_filename, result_img)
            print(f"Result image with annotations saved as '{result_filename}'")

        # Save warped labels if they exist
        if warped_labels:
            for idx, label in enumerate(warped_labels):
                label_filename = os.path.join(output_dir, f'warped_label_{idx}.png')
                cv2.imwrite(label_filename, label)
                print(f"Warped label saved as '{label_filename}'")

        # Clear memory
        del ref_image, transformed_image, transformed_blurred, result_img, warped_labels
        gc.collect()

    except Exception as e:
        print(f"Error processing image pair: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Label Transformer CLI')
    parser.add_argument('--input', type=str, required=True, help='Path to input folder containing original and generated images')
    parser.add_argument('--config', type=str, default='settings.yaml', help='Path to the configuration file (default: settings.yaml)')
    parser.add_argument('--ocr_engine', type=str, default='paddle', choices=['paddle', 'tesseract'], help='OCR engine to use (default: paddle)')
    
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print("Error: Input must be a directory containing 'original' and 'generated' subdirectories.")
        return

    label_transformer = LabelTransformer(ocr_engine=args.ocr_engine, config_path=args.config)

    input_folder = os.path.abspath(args.input)
    output_folder = input_folder + '_output'
    os.makedirs(output_folder, exist_ok=True)
    
    original_folder = os.path.join(input_folder, 'original')
    generated_folder = os.path.join(input_folder, 'generated')

    if not os.path.isdir(original_folder) or not os.path.isdir(generated_folder):
        print("Error: Input folder must contain 'original' and 'generated' subdirectories.")
        return

    original_images = glob.glob(os.path.join(original_folder, '*.[pj][np][g]'))  # matches .png, .jpg, .jpeg
    for original_path in original_images:
        image_name = os.path.splitext(os.path.basename(original_path))[0]
        generated_path = os.path.join(generated_folder, f"{image_name}.*")
        generated_files = glob.glob(generated_path)
        
        if not generated_files:
            print(f"Warning: No matching generated image found for {image_name}")
            continue
        
        generated_path = generated_files[0]  # Take the first match if multiple exist
        
        image_output_dir = os.path.join(output_folder, image_name)
        os.makedirs(image_output_dir, exist_ok=True)
        process_image_pair(label_transformer, original_path, generated_path, image_output_dir)
        
        gc.collect()

if __name__ == "__main__":
    main()
