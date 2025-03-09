import os
import glob
import random
import shutil
import zipfile
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main_prep(zip_path, output_dir, destination_folder):
    """ 
    Extract ZIP, create destination folder, merge images, 
    ensure uniform image sizes, and apply augmentation. 
    """

    if not os.path.exists(zip_path):
        print(f"Error: ZIP file '{zip_path}' not found!")
        return

    if not os.path.exists(output_dir):
        print("Extracting ZIP file...")
        extract_zip(zip_path, output_dir)
    else:
        print("ZIP file already extracted. Skipping this step.")

    os.makedirs(destination_folder, exist_ok=True)
    
    source_folder_1 = os.path.join(output_dir, "Testing")
    source_folder_2 = os.path.join(output_dir, "Training")
    
    class_folders = ["glioma", "meningioma", "notumor", "pituitary"]
    
    # Merge and limit images to 300 per class
    merge_and_limit_images(source_folder_1, source_folder_2, destination_folder, class_folders)
    
    # Check and resize images BEFORE augmentation
    check_and_resize_images(class_folders, destination_folder, target_size=(224, 224))

    # Augment images to reach 600 per class
    augment_images(class_folders, destination_folder, target_count=600)
    
    # Rename images after augmentation
    for class_folder in class_folders:
        rename_images_after_augmentation(os.path.join(destination_folder, class_folder))
    
    # Split data into train, test, and validation sets
    split_data(destination_folder)


def extract_zip(zip_path, extract_to):
    """ Extract the ZIP file if not already extracted. """
    
    if not os.path.exists(extract_to):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"ZIP file extracted to: {extract_to}")
    else:
        print(f"Directory already exists: {extract_to}. Skipping extraction.")

def merge_and_limit_images(source_folder_1, source_folder_2, destination_folder, class_folders):
    """ Merge images from two sources into a destination folder and limit each class to 300 images. """

    for class_folder in class_folders:
        src_class_path_1 = os.path.join(source_folder_1, class_folder)
        src_class_path_2 = os.path.join(source_folder_2, class_folder)
        dest_class_path = os.path.join(destination_folder, class_folder)

        os.makedirs(dest_class_path, exist_ok=True)

        for src_folder in [src_class_path_1, src_class_path_2]:
            if os.path.exists(src_folder):
                for img in glob.glob(os.path.join(src_folder, "*.*")):
                    filename = os.path.basename(img)
                    new_path = os.path.join(dest_class_path, filename)

                    counter = 1
                    while os.path.exists(new_path):
                        name, ext = os.path.splitext(filename)
                        new_filename = f"{name}_{counter}{ext}"
                        new_path = os.path.join(dest_class_path, new_filename)
                        counter += 1

                    shutil.move(img, new_path)

        images = sorted(glob.glob(os.path.join(dest_class_path, "*.*")))

        if len(images) > 300:
            excess_images = images[300:]
            for img in excess_images:
                os.remove(img)
                print(f"Deleted: {img}")

    print("Merging complete. Each class folder now contains at most 300 images.")
    
from PIL import Image

def check_and_resize_images(class_folders, destination_folder, target_size=(224, 224)):
    """
    Ensure all images in the dataset have the same size before augmentation.
    If any image has a different size, resize it to `target_size`.
    """
    for class_folder in class_folders:
        class_path = os.path.join(destination_folder, class_folder)
        images = glob.glob(os.path.join(class_path, "*.*"))

        for img_path in images:
            with Image.open(img_path) as img:
                if img.size != target_size:  # Check if image size is different
                    img_resized = img.resize(target_size)  # Resize to target size
                    img_resized.save(img_path)  # Overwrite the image with resized version
                    print(f"Resized: {img_path} to {target_size}")

    print("Image size check & resizing complete.")


def augment_images(class_folders, destination_folder, target_count=600):
    """ Augment images using TensorFlow without changing the color or size of the images. """
    
    datagen = ImageDataGenerator(
        rotation_range=30,  # Rotate the image within 30 degrees.
        horizontal_flip=True,  # Flip the image horizontally (mirror effect).
        fill_mode='nearest'  # Fill in missing pixels after transformation.
    )
    
    for class_folder in class_folders:
        class_path = os.path.join(destination_folder, class_folder)
        images = glob.glob(os.path.join(class_path, "*.*"))
        
        if not images:
            continue

        current_count = len(images)
        
        while current_count < target_count:
            img_path = random.choice(images)

            # Load the original image and get its dimensions
            image = tf.keras.preprocessing.image.load_img(img_path)
            orig_size = image.size  # Get original size (width, height)
            
            # Convert to array
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = image.reshape((1,) + image.shape)  # Reshape for generator
            
            for batch in datagen.flow(image, batch_size=1):
                aug_img = batch[0].astype('uint8')  # Convert back to uint8
                
                # Convert array back to image
                aug_img_pil = tf.keras.preprocessing.image.array_to_img(aug_img)
                
                # Resize back to original dimensions (prevents unwanted size changes)
                aug_img_pil = aug_img_pil.resize(orig_size)
                
                # Save the augmented image
                save_path = os.path.join(class_path, f"aug_{current_count}.jpg")
                aug_img_pil.save(save_path)
                
                current_count += 1
                if current_count >= target_count:
                    break



def rename_images_after_augmentation(class_path):
    """ Rename all images in the class directory after augmentation. """
    filelist = os.listdir(class_path)
    for i, file in enumerate(filelist):
        file_split = os.path.splitext(file)
        ext = file_split[1]
        new_name = f"{os.path.basename(class_path)}-{i+1}{ext}"
        old_path = os.path.join(class_path, file)
        new_path = os.path.join(class_path, new_name)
        os.rename(old_path, new_path)

def plot_sample_images(train_dir):
    """ Plot one image from each class in the train directory with class names. """
    
    class_folders = os.listdir(train_dir)  # List all class folders inside the train directory.
    fig, axes = plt.subplots(1, len(class_folders), figsize=(12, 6))  # Create subplots to show images.
    
    for ax, class_folder in zip(axes, class_folders):  # Loop over each class folder and corresponding subplot.
        class_path = os.path.join(train_dir, class_folder)  # Get the path for the current class folder.
        images = glob.glob(os.path.join(class_path, "*.*"))  # Get all image files in the current class folder.
        
        if images:  # If there are images in the folder:
            img_path = random.choice(images)  # Randomly choose one image.
            img = plt.imread(img_path)  # Read the image using matplotlib.
            ax.imshow(img)  # Show the image on the current axis (subplot).
            ax.set_title(class_folder)  # Set the title of the subplot to the class name.
            ax.axis("off")  # Hide axis labels for a cleaner look.
        else:
            ax.set_title(class_folder)  # If no images, still show the class name.
            ax.axis("off")  # Hide the axis.

    plt.show()  # Display the plot with all the images.


def split_data(destination_folder):
    """ Split the dataset into 70% training, 20% testing, and 10% validation. """
    train_dir = os.path.join(destination_folder, 'train')
    test_dir = os.path.join(destination_folder, 'test')
    val_dir = os.path.join(destination_folder, 'val')
    
    for directory in [train_dir, test_dir, val_dir]:
        os.makedirs(directory, exist_ok=True)
    
    class_folders = ["glioma", "meningioma", "notumor", "pituitary"]
    
    for class_folder in class_folders:
        class_path = os.path.join(destination_folder, class_folder)
        images = glob.glob(os.path.join(class_path, "*.*"))
        random.shuffle(images)
        
        train_split = int(0.7 * len(images))
        test_split = int(0.2 * len(images))
        
        train_images = images[:train_split]
        test_images = images[train_split:train_split + test_split]
        val_images = images[train_split + test_split:]

        for img_set, img_dir in zip([train_images, test_images, val_images], [train_dir, test_dir, val_dir]):
            class_dest = os.path.join(img_dir, class_folder)
            os.makedirs(class_dest, exist_ok=True)
            for img in img_set:
                shutil.move(img, os.path.join(class_dest, os.path.basename(img)))

    # After all images have been moved, delete empty class folders
    for class_folder in class_folders:
        class_path = os.path.join(destination_folder, class_folder)
        if not os.listdir(class_path):  # Check if the folder is empty
            os.rmdir(class_path)  # Remove empty folder
            print(f"Deleted empty folder: {class_path}")

if __name__ == "__main__":
    zip_path = r"C:\Eden's Project\archive(2).zip"
    output_dir = r"C:\Eden's Project\unzipped"
    destination_folder = r"C:\Eden's Project\processed_data"
    
    main_prep(zip_path, output_dir, destination_folder)
