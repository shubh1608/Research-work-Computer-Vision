import os

classification_base_path = "../../Image-Data-Collection/images/image_classification/"
classification_save_location = "../../Image-Data-Collection/images/processed_images/image_classification"

#use None when you dont want to limit number of images
per_class_image_read_number = 10

def read_class_images_generator(base_path):
    for dir in os.listdir(base_path):
        if dir == "mixed":
            #to-do - read mixed images
            continue
        images = load_images_from_folder(os.path.join(base_path, dir))
        yield (dir, images)
            
def load_images_from_folder(folder):
    images = []
    filenames = os.listdir(folder)

    if per_class_image_read_number != None:
        filenames = os.listdir(folder)[:per_class_image_read_number]

    for filename in filenames:
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append((filename, img))
    return images

def save_processed_images(dir, operation_id, processed_images):

    location = create_dir(dir, operation_id)

    for (filename, image) in processed_images:
        cv2.imwrite(os.path.join(location, filename), image)

def create_dir(dir, operation_id):
    folder_path = os.path.join(save_location, dir)
    location = None
    if not os.path.exists(folder_path):
        location = os.path.join(save_location, dir, str(operation_id))
        os.makedirs(location)
    else:
        location = os.path.join(folder_path, str(operation_id))
        os.mkdir(location)
    return location
