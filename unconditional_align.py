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
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False,)

    def detect_text_regions(self, img):
        """
        Detect text regions in the image using the selected OCR engine.
        """
        if self.ocr_engine == 'paddle':
            return self._detect_text_regions_paddle(img)
        else:
            return self._detect_text_regions_pytesseract(img)

    def _detect_text_regions_paddle(self, img):
        """
        Detect text regions using PaddleOCR.
        """
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
        """
        Detect text regions using Pytesseract.
        """
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

    # def display_text_regions(self, img, boxes):
    #     """
    #     Display the image with detected text regions and annotations.
    #     """
    #     display_img = img.copy()
    #     for box in boxes:
    #         x, y, w, h = box['x'], box['y'], box['w'], box['h']
    #         cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #         cv2.putText(display_img, box['text'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    #     # Downscale the image to 1/3rd of its original size
    #     display_img_small = cv2.resize(display_img, (display_img.shape[1] // 3, display_img.shape[0] // 3))
    #     cv2.imshow('Detected Text Regions', display_img_small)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    def stochastic_transform(self, image):
        """
        Apply stochastic transformations to the image.
        """
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if random.random() < self.config['stochastic_transform']['probabilities']['flip_xy']:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

            
        if random.random() < self.config['stochastic_transform']['probabilities']['rotate']:
            angle = random.uniform(*self.config['stochastic_transform']['rotation']['angle_range'])
            img = img.rotate(angle, expand=True, resample=Image.BICUBIC)
        
        if random.random() < self.config['stochastic_transform']['probabilities']['crop']:
            width, height = img.size
            crop_percent = random.uniform(*self.config['stochastic_transform']['crop']['percent_range'])
            new_width = int(width * crop_percent)
            new_height = int(height * crop_percent)
            left = random.randint(0, width - new_width)
            top = random.randint(0, height - new_height)
            img = img.crop((left, top, left + new_width, top + new_height))
        
        if random.random() < self.config['stochastic_transform']['probabilities']['color_shift']:
            r, g, b = img.split()
            shift_range = self.config['stochastic_transform']['color_shift']['range']
            r = r.point(lambda i: i + random.randint(*shift_range))
            g = g.point(lambda i: i + random.randint(*shift_range))
            b = b.point(lambda i: i + random.randint(*shift_range))
            img = Image.merge('RGB', (r, g, b))
        
        img = img.resize((image.shape[1], image.shape[0]), Image.BICUBIC)
        
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def transform_label(self, ref_image, transformed_image=None, save_warped_labels=None, overlay_labels=None):
        save_warped_labels = self.config['display']['save_warped_labels'] if save_warped_labels is None else save_warped_labels
        overlay_labels = self.config['display']['overlay_labels'] if overlay_labels is None else overlay_labels

        print(f"transform_label called with:")
        print(f"  ref_image shape: {ref_image.shape}")
        
        if transformed_image is None:
            print("Generating transformed image...")
            transformed_image = self.stochastic_transform(ref_image)
            cv2.imwrite('image_b.png', transformed_image)
            print("Transformed image saved as 'image_b.png'")
        else:
            print(f"  transformed_image shape: {transformed_image.shape}")
        
        print(f"  save_warped_labels: {save_warped_labels}")
        print(f"  overlay_labels: {overlay_labels}")

        # Detect text regions in ref_image
        text_regions = self.detect_text_regions(ref_image)
        if text_regions is None:
            print("Could not detect labels via OCR.")
            return None, None

        # Display detected text regions on ref_image
        #self.display_text_regions(ref_image, text_regions)

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

def apply_mosaic_blur(image, box, block_size=20, blur_percentage=0.6):
    x, y, w, h = cv2.boundingRect(box)
    
    # Calculate the dimensions of the central 60% area
    center_w = int(w * blur_percentage)
    center_h = int(h * blur_percentage)
    start_x = x + (w - center_w) // 2
    start_y = y + (h - center_h) // 2
    
    # Extract the central region
    roi = image[start_y:start_y+center_h, start_x:start_x+center_w]
    
    # Resize the ROI to a smaller size
    small = cv2.resize(roi, (block_size, block_size), interpolation=cv2.INTER_LINEAR)
    
    # Resize the small image back to the original size
    blurred = cv2.resize(small, (center_w, center_h), interpolation=cv2.INTER_NEAREST)
    
    # Put the blurred region back into the image
    image[start_y:start_y+center_h, start_x:start_x+center_w] = blurred
    return image

def resize_image(image, max_size=1500):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def process_image(label_transformer, ref_image_path, output_dir):
    try:
        ref_image = cv2.imread(ref_image_path)
        if ref_image is None:
            print(f"Error: Unable to load the reference image from {ref_image_path}")
            return

        print(f"Processing image: {ref_image_path}")
        print(f"Original Reference Image shape: {ref_image.shape}")

        # Resize the image if it's too large
        ref_image = resize_image(ref_image)
        print(f"Resized Reference Image shape: {ref_image.shape}")

        # Generate transformed image
        transformed_image = label_transformer.stochastic_transform(ref_image)
        
        # Save transformed image without annotations
        transformed_image_path = os.path.join(output_dir, 'transformed_image.png')
        cv2.imwrite(transformed_image_path, transformed_image)
        print(f"Transformed image saved as '{transformed_image_path}'")

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

        # Save the transformed image with blurred text regions
        transformed_blurred_path = os.path.join(output_dir, 'transformed_blurred.png')
        cv2.imwrite(transformed_blurred_path, transformed_blurred)
        print(f"Transformed image with blurred text regions saved as '{transformed_blurred_path}'")

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
        print(f"Error processing image {ref_image_path}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Label Transformer CLI')
    parser.add_argument('--input', type=str, required=True, help='Path to input folder containing images')
    parser.add_argument('--config', type=str, default='settings.yaml', help='Path to the configuration file (default: settings.yaml)')
    parser.add_argument('--ocr_engine', type=str, default='paddle', choices=['paddle', 'tesseract'], help='OCR engine to use (default: paddle)')
    
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print("Error: Input must be a directory containing images.")
        return

    label_transformer = LabelTransformer(ocr_engine=args.ocr_engine, config_path=args.config)

    input_folder = os.path.abspath(args.input)
    output_folder = input_folder + '_siftout'
    os.makedirs(output_folder, exist_ok=True)
    
    image_files = glob.glob(os.path.join(input_folder, '*.[pj][np][g]'))  # matches .png, .jpg, .jpeg
    for image_file in image_files:
        image_name = os.path.splitext(os.path.basename(image_file))[0]
        image_output_dir = os.path.join(output_folder, image_name)
        os.makedirs(image_output_dir, exist_ok=True)
        process_image(label_transformer, image_file, image_output_dir)
        # Clear memory after each image
        gc.collect()

if __name__ == "__main__":
    main()
