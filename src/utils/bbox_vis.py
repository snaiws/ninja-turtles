import cv2
import glob
import os
from dotenv import load_dotenv

# Load the .env file from the specified path
load_dotenv('/home/alienattack/workspace/local_workspace/01_Projects/project2_CV-AD/EDA_test/ninja-turtles/.env')

# Get the dataset and model directories from the .env file
dataset_dir = os.getenv('DIR_DATA')

# Read class names from classes.txt
def load_class_names(label_dir):
    class_file = os.path.join(label_dir, 'classes.txt')
    class_names = []
    if os.path.exists(class_file):
        with open(class_file, 'r') as f:
            class_names = f.read().splitlines()
    return class_names

def draw_bounding_boxes(image_path, label_path, class_names):
    # Step 1: Load the image
    img = cv2.imread(image_path)
    img_h, img_w, _ = img.shape  # Get the image dimensions

    # Step 2: Read and parse the label file
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # Step 3: Iterate through each label and draw a bounding box
    for line in lines:
        label = line.split()
        class_id = int(label[0])  # Class ID
        norm_x = float(label[1])  # Normalized center X coordinate
        norm_y = float(label[2])  # Normalized center Y coordinate
        norm_w = float(label[3])  # Normalized width
        norm_h = float(label[4])  # Normalized height

        # Convert normalized coordinates to pixel values
        box_w = int(norm_w * img_w)
        box_h = int(norm_h * img_h)
        center_x = int(norm_x * img_w)
        center_y = int(norm_y * img_h)

        # Calculate the top-left and bottom-right coordinates of the rectangle
        top_left_x = int(center_x - box_w / 2)
        top_left_y = int(center_y - box_h / 2)
        bottom_right_x = int(center_x + box_w / 2)
        bottom_right_y = int(center_y + box_h / 2)

        # Step 4: Draw the rectangle on the image
        cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

        # Map class ID to class name using the correct index
        if class_id < len(class_names):
            class_name = class_names[class_id]  # Use zero-indexed class names
        else:
            class_name = "Unknown"
        
        # Draw class name near the bounding box
        cv2.putText(img, class_name, (top_left_x, top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Step 5: Display the image with bounding boxes
    cv2.namedWindow("Image with Bounding Boxes", cv2.WINDOW_NORMAL)
     # Get screen size for centering
    screen_width = 1920  # Example: Adjust based on your screen resolution
    screen_height = 1080  # Example: Adjust based on your screen resolution

    # Move the window to the center of the screen
    window_x = int((screen_width - img_w) / 2)
    window_y = int((screen_height - img_h) / 2)
    cv2.moveWindow("Image with Bounding Boxes", window_x, window_y)
    
    # Resize window to a larger size for better observation
    cv2.resizeWindow("Image with Bounding Boxes", 1024, 768)
    
    # Display the image
    cv2.imshow("Image with Bounding Boxes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_images_for_set(set_type, num_examples=None):
    """
    Processes images for a specific dataset set (train, test, or val)
    :param set_type: 'train', 'test', or 'val'
    :param num_examples: The number of images to process. If None, process all images.
    """
    # Set paths for the selected set (train, test, or val)
    image_dir = os.path.join(dataset_dir, set_type, 'images')
    label_dir = os.path.join(dataset_dir, set_type, 'label') if set_type != 'test' else None

    # Load class names from the classes.txt file
    class_names = load_class_names(label_dir) if label_dir else []

    # Get all images from the directory
    image_files = glob.glob(os.path.join(image_dir, '*.jpg'))
    image_files.sort()  # Sort to have a consistent order

    # Ensure num_examples doesn't exceed the number of available images
    if num_examples is not None:
        num_examples = min(num_examples, len(image_files))
        image_files = image_files[:num_examples]

    # Loop through each image and find the corresponding label
    for image_file in image_files:
        if set_type != 'test':
            # For 'train' and 'val' sets, labels are available
            label_file = os.path.join(label_dir, os.path.basename(image_file).replace('.jpg', '.txt'))
            if os.path.exists(label_file):
                # Call the function to draw bounding boxes for this pair of image and label
                print(f"Processing {image_file} with label {label_file}")
                draw_bounding_boxes(image_file, label_file, class_names)
            else:
                print(f"Label file for {image_file} not found: {label_file}")
        else:
            # For 'test' set, only process the images (no labels)
            print(f"Processing test image {image_file}")
            img = cv2.imread(image_file)
            cv2.imshow("Test Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# Example usage:
# Process 5 images from the training set
process_images_for_set('train', num_examples=5)
