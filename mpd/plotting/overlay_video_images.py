import os

import cv2
import numpy as np
from PIL import Image


def extract_last_frame(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return None

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set the frame position to the last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

    # Read the last frame
    ret, last_frame = cap.read()

    # Release the video capture object
    cap.release()

    if ret:
        return last_frame
    else:
        print(f"Error reading last frame from {video_path}")
        return None


def process_videos_extract_last_frame(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith((".mp4", ".avi", ".mov")):  # Add more video extensions if needed
            video_path = os.path.join(folder_path, filename)
            last_frame = extract_last_frame(video_path)

            if last_frame is not None:
                # Save the last frame as an image
                output_filename = f"last_frame_{os.path.splitext(filename)[0]}.png"
                output_path = os.path.join(folder_path, output_filename)
                cv2.imwrite(output_path, last_frame)
                print(f"Saved last frame of {filename} as {output_filename}")


def crop_image(image, crop_params):
    """Crop the image based on the given parameters."""
    x, y, w, h = crop_params
    return image[y : y + h, x : x + w]


def create_subtle_overlay(folder_path, output_filename="subtle_overlay.png", overlay_strength=0.3, crop_params=None):
    # Get all image files
    image_files = [f for f in os.listdir(folder_path) if f.startswith("last_frame_") and f.endswith((".jpg", ".png"))]

    if not image_files:
        print("No extracted last frames found.")
        return

    # Read the first image to get dimensions
    first_image = cv2.imread(os.path.join(folder_path, image_files[0]))

    # Crop the first image if crop parameters are provided
    if crop_params:
        first_image = crop_image(first_image, crop_params)

    height, width = first_image.shape[:2]

    # Create a blank canvas for the overlay
    overlay = np.zeros((height, width, 3), dtype=np.float32)

    # Loop through all images and add them to the overlay
    for i, filename in enumerate(image_files):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        # Crop the image if crop parameters are provided
        if crop_params:
            img = crop_image(img, crop_params)
        else:
            # Ensure the image has the same dimensions as the first one
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))

        # Add the image to the overlay with reduced opacity
        if i == 0:
            # Use the first image as the base
            overlay = img.astype(np.float32)
        else:
            # Blend subsequent images
            overlay = cv2.addWeighted(overlay, 1 - overlay_strength, img.astype(np.float32), overlay_strength, 0)

    # Normalize the overlay image
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Convert from BGR to RGB
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Save the overlay image using PIL for better quality
    Image.fromarray(overlay_rgb).save(os.path.join(folder_path, output_filename))
    print(f"Subtle overlay image saved as {output_filename}")

    # Display some statistics
    print(f"Number of frames overlaid: {len(image_files)}")
    print(f"Overlay dimensions: {width}x{height}")
    print(f"Overlay strength: {overlay_strength}")
    if crop_params:
        print(
            f"Crop parameters: x={crop_params[0]}, y={crop_params[1]}, width={crop_params[2]}, height={crop_params[3]}"
        )


def create_enhanced_overlay(
    folder_path=None,
    output_filename="enhanced_overlay.png",
    base_weight=0.5,
    image_weight=0.5,
    crop_params=None,
    image_files_l=None,
    output_dir=None,
):
    # Get all image files
    if image_files_l is None:
        image_files = [
            f for f in os.listdir(folder_path) if f.startswith("last_frame_") and f.endswith((".jpg", ".png"))
        ]
    else:
        image_files = image_files_l

    if len(image_files) == 0:
        print("No extracted last frames found.")
        return

    # Read and process all images
    processed_images = []
    for filename in image_files:
        if image_files_l is not None:
            img_path = filename
        else:
            img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        if crop_params:
            img = crop_image(img, crop_params)

        processed_images.append(img)

    # Use the first image to set dimensions
    height, width = processed_images[0].shape[:2]

    # Create the base overlay using the first image
    overlay = processed_images[0].astype(np.float32)

    # Blend subsequent images
    for img in processed_images[1:]:
        # Ensure the image has the same dimensions
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))

        # Convert to float32 for calculations
        img = img.astype(np.float32)

        # Compute the difference between the current image and the overlay
        diff = cv2.absdiff(img, overlay)

        # Normalize the difference to enhance contrast
        diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

        # Blend the normalized difference with the overlay
        overlay = cv2.addWeighted(overlay, base_weight, diff_normalized, image_weight, 0)

    # Normalize the final overlay
    overlay = cv2.normalize(overlay, None, 0, 255, cv2.NORM_MINMAX)
    overlay = overlay.astype(np.uint8)

    # Convert from BGR to RGB
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Save the overlay image using PIL for better quality
    Image.fromarray(overlay_rgb).save(os.path.join(folder_path if output_dir is None else output_dir, output_filename))
    print(f"Enhanced overlay image saved as {output_filename}")

    # Display some statistics
    print(f"Number of frames overlaid: {len(image_files)}")
    print(f"Overlay dimensions: {width}x{height}")
    print(f"Base weight: {base_weight}, Image weight: {image_weight}")
    if crop_params:
        print(
            f"Crop parameters: x={crop_params[0]}, y={crop_params[1]}, width={crop_params[2]}, height={crop_params[3]}"
        )
