import os
import shutil
import random
import glob

# --- Configuration ---
# The folder where your original, unsorted images are (e.g., 'images/')
# This should point to the directory containing breed subfolders (e.g., 'Airedale', 'Basenji', etc.)
SOURCE_DIR = 'C:/Users/Jessi W/PycharmProjects/ImageClassifier/Assets/Assets/Images'

# The top-level folder where your 'training' and 'validation' folders will be created
OUTPUT_DIR = 'C:/Users/Jessi W/PycharmProjects/ImageClassifier/dataset'

# The ratio for the split (e.g., 0.8 means 80% for training, 20% for validation)
SPLIT_RATIO = 0.8

# --- End of Configuration ---

def create_dataset_structure(base_dir, classes):
    """Creates the necessary training and validation folders within the base_dir."""
    print("Creating directory structure...")
    os.makedirs(base_dir, exist_ok=True)
    for folder_type in ['training', 'validation']:
        type_path = os.path.join(base_dir, folder_type)
        os.makedirs(type_path, exist_ok=True)
        for class_name in classes:
            class_path = os.path.join(type_path, class_name)
            os.makedirs(class_path, exist_ok=True)
    print("Directory structure created.")


def main():
    """
    Finds dog breed classes from subfolder names, creates the dataset folder structure,
    and splits the images into training and validation sets.
    """
    if not os.path.isdir(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' not found.")
        print("Please ensure the Stanford Dogs dataset images are unzipped with breed subfolders.")
        return

    # --- NEW LOGIC FOR FINDING CLASSES AND IMAGE PATHS ---
    dog_breeds_found = set()
    all_image_paths = []
    class_image_map = {} # To store lists of image paths per class

    print(f"Scanning '{SOURCE_DIR}' for dog breed subfolders and images...")
    # os.walk allows us to traverse directory trees
    for root, dirs, files in os.walk(SOURCE_DIR):
        # We are interested in the immediate subdirectories of SOURCE_DIR,
        # as these are the breed names.
        # This check ensures we only process the direct subfolders of SOURCE_DIR as breeds.
        # It also handles cases where there might be hidden files/folders or other structures.
        if root == SOURCE_DIR:
            # 'dirs' here are the breed folders
            for breed_name in dirs:
                # Ensure it's a directory and not some other file
                breed_path = os.path.join(root, breed_name)
                if os.path.isdir(breed_path):
                    dog_breeds_found.add(breed_name)
                    class_image_map[breed_name] = [] # Initialize list for this breed

        # Collect image paths for the current breed subfolder
        # The 'root' will be 'SOURCE_DIR/BreedName'
        # The 'files' will be the images within that 'BreedName' folder
        breed_name_from_path = os.path.basename(root)
        if breed_name_from_path in dog_breeds_found: # Only process if it's a known breed folder
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg')): # Case-insensitive check for common image extensions
                    image_path = os.path.join(root, file)
                    all_image_paths.append(image_path)
                    class_image_map[breed_name_from_path].append(image_path)

    if not dog_breeds_found:
        print(f"Error: No dog breed subfolders found directly in '{SOURCE_DIR}'.")
        print("Expected structure: SOURCE_DIR/BreedName/image.jpg")
        return
    if not all_image_paths:
        print(f"Error: No '.jpg' or '.jpeg' images found in the breed subfolders of '{SOURCE_DIR}'.")
        return

    # Convert set to a sorted list for consistent processing
    dog_breeds = sorted(list(dog_breeds_found))
    print(f"Found {len(dog_breeds)} unique dog breed folders.")
    print(f"Dog breed folders: {dog_breeds}")
    print(f"Total images found: {len(all_image_paths)}")

    # 1. Create the required folder structure for training and validation
    create_dataset_structure(OUTPUT_DIR, dog_breeds)

    # 2. Split and copy files for each class (dog breed)
    print("\nSplitting and copying files for each dog breed...")
    for breed_name in dog_breeds:
        breed_files = class_image_map.get(breed_name, []) # Get files for this breed
        if not breed_files:
            print(f"  Warning: No images found for breed '{breed_name}'. Skipping.")
            continue

        random.shuffle(breed_files) # Shuffle for random split

        split_index = int(len(breed_files) * SPLIT_RATIO)
        train_files = breed_files[:split_index]
        val_files = breed_files[split_index:]

        # Copy training files to their respective training breed subfolder
        for f_path in train_files:
            destination_path = os.path.join(OUTPUT_DIR, 'training', breed_name, os.path.basename(f_path))
            shutil.copy(f_path, destination_path)

        # Copy validation files to their respective validation breed subfolder
        for f_path in val_files:
            destination_path = os.path.join(OUTPUT_DIR, 'validation', breed_name, os.path.basename(f_path))
            shutil.copy(f_path, destination_path)

        print(f"  - Processed breed '{breed_name}': {len(train_files)} training, {len(val_files)} validation files.")

    print("\nDataset successfully organized into training and validation folders!")
    print(f"Check the structure in the '{OUTPUT_DIR}' directory.")


if __name__ == '__main__':
    main()