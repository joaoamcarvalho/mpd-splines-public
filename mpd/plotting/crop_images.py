import cv2
import os
import argparse


def center_crop(image_path, width, height, output_path=None):
    """
    Crop an image from the center using OpenCV.

    Args:
        image_path (str): Path to the input image
        width (int): Width of crop region
        height (int): Height of crop region
        output_path (str, optional): Path to save the cropped image

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read the image")

        # Get image dimensions
        img_height, img_width = image.shape[:2]

        # Validate crop dimensions
        if width <= 0 or height <= 0:
            raise ValueError("Crop dimensions must be positive")
        if width > img_width or height > img_height:
            raise ValueError("Crop dimensions exceed image size")

        # Calculate center coordinates
        x_center = img_width // 2
        y_center = img_height // 2

        # Calculate crop coordinates
        x1 = x_center - (width // 2)
        y1 = y_center - (height // 2)
        x2 = x1 + width
        y2 = y1 + height

        # Adjust coordinates if they go outside image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)

        # Perform the crop
        cropped_image = image[y1:y2, x1:x2]

        # Generate output path if not provided
        if output_path is None:
            file_name, file_ext = os.path.splitext(image_path)
            output_path = f"{file_name}_center_cropped{file_ext}"

        # Save the cropped image
        cv2.imwrite(output_path, cropped_image)
        print(f"Center-cropped image saved to: {output_path}")
        return True

    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def crop_image(image_path, x, y, width, height, output_path=None):
    """
    Crop an image from specified coordinates using OpenCV.

    Args:
        image_path (str): Path to the input image
        x (int): X-coordinate of top-left corner of crop region
        y (int): Y-coordinate of top-left corner of crop region
        width (int): Width of crop region
        height (int): Height of crop region
        output_path (str, optional): Path to save the cropped image

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read the image")

        # Get image dimensions
        img_height, img_width = image.shape[:2]

        # Validate crop parameters
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("Crop parameters must be positive")
        if x + width > img_width or y + height > img_height:
            raise ValueError("Crop region exceeds image dimensions")

        # Perform the crop
        cropped_image = image[y : y + height, x : x + width]

        # Generate output path if not provided
        if output_path is None:
            file_name, file_ext = os.path.splitext(image_path)
            output_path = f"{file_name}_cropped{file_ext}"

        # Save the cropped image
        cv2.imwrite(output_path, cropped_image)
        print(f"Cropped image saved to: {output_path}")
        return True

    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Crop an image using OpenCV")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--center", action="store_true", help="Use center cropping")
    parser.add_argument("--width", type=int, required=True, help="Width of crop region")
    parser.add_argument("--height", type=int, required=True, help="Height of crop region")
    parser.add_argument("--x", type=int, help="X-coordinate of crop region (for manual crop)")
    parser.add_argument("--y", type=int, help="Y-coordinate of crop region (for manual crop)")
    parser.add_argument("--output", "-o", help="Path to save the cropped image")

    args = parser.parse_args()

    if args.center:
        success = center_crop(args.image_path, args.width, args.height, args.output)
    else:
        if args.x is None or args.y is None:
            print("Error: x and y coordinates required for manual cropping")
            exit(1)
        success = crop_image(args.image_path, args.x, args.y, args.width, args.height, args.output)

    if not success:
        print("Failed to crop image")
        exit(1)


if __name__ == "__main__":
    main()
