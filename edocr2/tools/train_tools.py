import random, time, os, math, cv2
import numpy as np
from collections import Counter

############ Synthetic Generation ###############################################
def get_and_process_fonts(dir_target, alphabet):

    def font_supports_alphabet(font_path, alphabet):
        from PIL import ImageFont
        """
        Check if the given font supports all characters in the specified alphabet.

        :param font_path: Path to the font file.
        :param alphabet: A string containing all the characters of the alphabet to check.
        :return: True if the font supports the entire alphabet, False otherwise.
        """
        try:
            # Load the font
            font = ImageFont.truetype(font_path, size=10)  # Font size is arbitrary
        except IOError:
            print(f"Error: Cannot load font from {font_path}")
            return False

        # Check each character in the alphabet
        try:
            for char in alphabet:
                if not font.getmask(char).getbbox():
                    # If getbbox returns None, the character is not supported
                    print(f"Character '{char}' is not supported by {font_path}")
                    return False
        except:
            return False

        return True

    def move_file_to_directory(file_path, target_directory):
        """
        Move a file to a new directory.
        
        :param file_path: The path to the file that will be moved.
        :param target_directory: The directory where the file will be moved.
        """
        try:
            # Ensure the target directory exists
            if not os.path.exists(target_directory):
                os.makedirs(target_directory)

            # Move the file
            shutil.move(file_path, target_directory)
            print(f"Moved: {file_path} -> {target_directory}")

        except Exception as e:
            print(f"Error moving {file_path} to {target_directory}: {e}")
    
    #Download files from keras_ocr:
    from edocr2.keras_ocr.tools import download_and_verify
    import glob, zipfile, shutil
    fonts_zip_path = download_and_verify(
        url="https://github.com/faustomorales/keras-ocr/releases/download/v0.8.4/fonts.zip",
        sha256="d4d90c27a9bc4bf8fff1d2c0a00cfb174c7d5d10f60ed29d5f149ef04d45b700",
        filename="fonts.zip",
        cache_dir='.',
    )
    fonts_dir = os.path.join('.', "fonts")
    if len(glob.glob(os.path.join(fonts_dir, "**/*.ttf"))) != 2746:
        print("Unzipping fonts ZIP file.")
        with zipfile.ZipFile(fonts_zip_path) as zfile:
            zfile.extractall(fonts_dir)

    for root, dirs, _ in os.walk('fonts'):
        for dir in dirs:
            for _, _, files2 in os.walk(os.path.join(root, dir)):
                for file in files2:
                    if file.endswith("Regular.ttf"):
                        font_path = os.path.join(root, dir, file)
                        if font_supports_alphabet(font_path, alphabet):
                            # Add the full path to the array
                            move_file_to_directory(font_path, dir_target)
    shutil.rmtree('fonts')

def get_balanced_text_generator(alphabet, string_length=(5, 10), lowercase=False):
    '''
    Generates batches of sentences ensuring perfectly balanced symbol distribution.
    Args:
        alphabet: string of characters
        batch_size: number of sentences per batch
        string_length: tuple defining range of sentence length
        lowercase: convert alphabet to lowercase
    Return:
        list of sentence strings
    '''
    # Initialize a counter to track the number of times each character is used
    symbol_counter = Counter({char: 0 for char in alphabet})
    
    while True:
        # Calculate the total number of generated symbols
        total_generated = sum(symbol_counter.values())

        # Adjust probabilities to balance the frequency of each symbol
        weights = {char: total_generated - count + 1 for char, count in symbol_counter.items()}
        total_weight = sum(weights.values())
        probabilities = [weights[char] / total_weight for char in alphabet]

        # Sample a sentence based on the adjusted probabilities
        sentence = random.choices(alphabet, weights=probabilities, k=random.randint(string_length[0], string_length[1]))
        sentence = "".join(sentence)

        # Update the symbol counter
        symbol_counter.update(sentence)

        if lowercase:
            sentence = sentence.lower()
        
        yield sentence

def get_backgrounds(height, width, samples):
    backgrounds = []
    backg_path = os.path.join(os.getcwd(), 'edocr2/tools/backgrounds')
    backg_files = os.listdir(backg_path)
    for _ in range(samples):
        backg_file = random.choice(backg_files)
        img = cv2.imread(os.path.join(backg_path, backg_file))
        y, x = random.randint(0, img.shape[0] - height), random.randint(0, img.shape[1] - width)
        backg = img[y : y + height, x : x + width][:]
        backgrounds.append(backg)

    return backgrounds

def filter_wrong_samples(generator, white_pixel_threshold=0.05):
    """A generator wrapper that filters out samples with too many white pixels.
    
    Args:
    generator: The original generator that produces image samples.
    white_pixel_threshold: The maximum allowed ratio of white pixels.
    
    Yields:
    Valid samples that meet the white pixel threshold criteria.
    """
    for image, text in generator:
        # Convert image to grayscale to count white pixels
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold to create a binary image (white pixels = 255, other = 0)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

        # Calculate total pixels and the number of white pixels
        total_pixels = binary_image.size
        white_pixels = np.sum(binary_image == 255)
        
        # Calculate the percentage of white pixels
        white_pixel_ratio = white_pixels / total_pixels
        
        # Yield the sample only if the white pixel ratio is within the acceptable threshold
        if white_pixel_ratio >= white_pixel_threshold:
            yield cv2.bitwise_not(image), text
        '''else:
            print(f"Skipping sample due to low white pixel ratio ({white_pixel_ratio:.2%})")'''

def generate_drawing_imgs(image_gen_params, backgrounds):

    def choose_background_no_overlap(text_img, background_list):
        """Choose a random background from the list that doesn't overlap with the white text.
        
        Args:
        text_img: A binary image where the text is white (255) on black (0).
        background_list: A list of background images.
        
        Returns:
        The chosen background image that doesn't overlap with the white text.
        """
        def check_overlap(text_img, background_img):
            """Check if there is an overlap between black pixels of the background and white pixels of the text.
            
            Args:
            text_img: A binary image where the text is white (255) on black (0).
            background_img: A grayscale or RGB background image.
            
            Returns:
            bool: True if there is an overlap, False otherwise.
            """
            # Ensure both images are of the same size
            if text_img.shape != background_img.shape[:2]:
                raise ValueError("Text image and background image must have the same dimensions")
            
            # Convert background to grayscale if it's RGB
            if len(background_img.shape) == 3:
                background_gray = cv2.cvtColor(background_img, cv2.COLOR_BGR2GRAY)
            else:
                background_gray = background_img

            # Identify where the text image has white pixels (text pixels)
            text_mask = text_img == 0

            # Identify where the background has black pixels (0 value)
            background_black_mask = background_gray == 0
            
            # Check if any black background pixels overlap with the white text pixels
            overlap = np.any(np.logical_and(text_mask, background_black_mask))

            return overlap

        # Try picking a background that does not overlap
        for _ in range(len(background_list)*2):  # Limit the number of retries to avoid infinite loops
            background_img = random.choice(background_list)
            if not check_overlap(text_img, background_img):
                return background_img

    def apply_text_on_background(text_img, text_binary, background_img):
        """Apply the text image over the background, assuming no overlap."""
        # Create a mask where text_binary is white (255), ndicating text
        text_mask = text_binary == 0
        
        # Create a copy of background_img to avoid modifying the original image
        result = background_img.copy()

        inverted_text_img = cv2.bitwise_not(text_img)
        result[text_mask] = inverted_text_img[text_mask]

        return result

    def compact_bounding_box(box_group):
        from edocr2.tools.ocr_pipelines import group_polygons_by_proximity
        box_groups = []
        for b in box_group:
            for xy, _ in b:
                box_groups.append(xy)
                    
        box_groups = group_polygons_by_proximity(box_groups, eps = 10)
        
        dummy_char = '1'
        dummy_box_groups = []

        for box in box_groups:
            dummy_box_groups.append([(np.array(box).astype(np.int32), dummy_char)])

        return dummy_box_groups
    
    from edocr2.keras_ocr import data_generation
    """Generate images with text on a background, ensuring no overlap."""
    while True:
        # Create image generators for training and validation
        image_generator_train = data_generation.get_image_generator(**image_gen_params)
        text_image, lines = next(image_generator_train)

        # Convert text image to binary
        _, binary_text_img = cv2.threshold(cv2.cvtColor(text_image, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY_INV)

        # Choose a background that doesn't overlap with the text
        background = choose_background_no_overlap(binary_text_img, backgrounds)
        if background is not None:
            # Apply the text image on the background
            image = apply_text_on_background(text_image, binary_text_img, background)
            lines = compact_bounding_box(lines)
            yield image, lines

def save_recog_samples(alphabet, fonts, samples, recognizer, save_path = './recog_samples'):
    """Generate and save a few samples along with their labels.
    
    Args:
    recognizer: The recognizer model (trained or not).
    image_generator: The generator to produce the images.
    sample_count: Number of samples to generate.
    save_path: Path where the samples will be saved.
    """
    from edocr2.keras_ocr import data_generation

    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    # Generate and save the samples
    for i in range(samples):

        text_generator = get_balanced_text_generator(alphabet, (5, 10))

        image_gen_params = {
        'height': 256,
        'width': 256,
        'text_generator': text_generator,
        'font_groups': {alphabet: fonts},  # Use all fonts
        'font_size': (20, 40),
        'margin': 10,
        }

        # Create image generators for training and validation
        image_generators_train = data_generation.get_image_generator(**image_gen_params)

        # Helper function to convert image generators to recognizer input
        def convert_generators(image_generators):
            return data_generation.convert_image_generator_to_recognizer_input(
                    image_generator=image_generators,
                    max_string_length=min(recognizer.training_model.input_shape[1][1], 10),
                    target_width=recognizer.model.input_shape[2],
                    target_height=recognizer.model.input_shape[1],
                    margin=1) 

        # Convert training and validation image generators
        recog_img_gen_train = convert_generators(image_generators_train)
        filter_gen = filter_wrong_samples(recog_img_gen_train, white_pixel_threshold=0.05)
        image, text = next(filter_gen)
        
        # Save the image
        image_filename = os.path.join(save_path, f'{i + 1}.png')
        cv2.imwrite(image_filename, image)
        
        # Save the label in a text file
        label_filename = os.path.join(save_path, f'{i + 1}.txt')
        with open(label_filename, 'w') as label_file:
            label_file.write(text)

def save_detect_samples(alphabet, fonts, samples, save_path = './detect_samples'):
    
    os.makedirs(save_path, exist_ok=True)

    text_generator = get_balanced_text_generator(alphabet, (5, 10))
    height, width = 640, 640
    backgrounds = get_backgrounds(height, width, samples)

    image_gen_params = {
    'height': height,
    'width': width,
    'text_generator': text_generator,
    'font_groups': {alphabet: fonts},  # Use all fonts
    'font_size': (20, 80),
    'margin': 25,
    'rotationZ': (-90, 120)
    }

    image_gen = generate_drawing_imgs(image_gen_params, backgrounds)
    for i in range(samples):
        image, lines = next(image_gen)

        # Save the image
        image_filename = os.path.join(save_path, f'img_{i + 1}.png')
        cv2.imwrite(image_filename, image)

        label_filename = os.path.join(save_path, f'gt_img_{i + 1}.txt')
        label = ''

        for box in lines:
            for xy, _ in box:
                for vertex in xy:
                    label += str(int(vertex[0])) + ', ' + str(int(vertex[1])) + ', '
                #pts=np.array([(xy[0]),(xy[1]),(xy[2]),(xy[3])], dtype=np.int32).reshape((-1, 1, 2))
                #cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
                label += '### \n'

        with open(label_filename, 'w') as txt_file:
            txt_file.write(label)

        #cv2.imshow('Image with Oriented Bounding Box', image)
        #cv2.waitKey(0)  # Wait for a key press to close the image
        #cv2.destroyAllWindows()

############ Synthetic Training ################################################

def train_synth_recognizer(alphabet, fonts, pretrained = None, samples = 1000, batch_size = 256, epochs = 10, string_length = (5, 10), basepath = os.getcwd(), val_split = 0.2):
    '''Starts the training of the recognizer on generated data.
    Args:
    alphabet: string of characters
    backgrounds: list of backgrounds images
    fonts: list of fonts with format *.ttf
    batch_size: batch size for training
    recognizer_basepath: desired path to recognizer
    pretrained_model: path to pretrained weights

    '''
    import tensorflow as tf
    from edocr2 import keras_ocr
    basepath = os.path.join(basepath,
    f'recognizer_{time.gmtime(time.time()).tm_hour}'+f'_{time.gmtime(time.time()).tm_min}')

    text_generator = get_balanced_text_generator(alphabet, string_length)

    image_gen_params = {
    'height': 256,
    'width': 256,
    'text_generator': text_generator,
    'font_groups': {alphabet: fonts},  # Use all fonts
    'font_size': (20, 40),
    'margin': 10
    }

    # Create image generators for training and validation
    image_generators_train = keras_ocr.data_generation.get_image_generator(**image_gen_params)
    image_generators_val = keras_ocr.data_generation.get_image_generator(**image_gen_params)
    
    recognizer = keras_ocr.recognition.Recognizer(alphabet=alphabet)
    if pretrained:
        recognizer.model.load_weights(pretrained)
    recognizer.compile()
    #for layer in recognizer.backbone.layers:
     #   layer.trainable = False

    # Helper function to convert image generators to recognizer input
    def convert_generators(image_generators):
        return keras_ocr.data_generation.convert_image_generator_to_recognizer_input(
                image_generator=image_generators,
                max_string_length=min(recognizer.training_model.input_shape[1][1], string_length[1]),
                target_width=recognizer.model.input_shape[2],
                target_height=recognizer.model.input_shape[1],
                margin=1) 

    # Convert training and validation image generators
    recog_img_gen_train = filter_wrong_samples(convert_generators(image_generators_train))
    recog_img_gen_val = filter_wrong_samples(convert_generators(image_generators_val))

    recognition_train_generator = recognizer.get_batch_generator(recog_img_gen_train, batch_size)
    recognition_val_generator = recognizer.get_batch_generator(recog_img_gen_val, batch_size)

    recognizer.training_model.fit(
        recognition_train_generator,
        epochs=epochs,
        steps_per_epoch=math.ceil((1 - val_split) * samples / batch_size),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=5),
            tf.keras.callbacks.CSVLogger(f'{basepath}.csv', append=True),
            tf.keras.callbacks.ModelCheckpoint(filepath=f'{basepath}.keras',save_best_only=True),
        ],
        validation_data=recognition_val_generator,
        validation_steps=math.ceil(val_split * samples / batch_size),
    )
    return basepath

def train_synth_detector(alphabet, fonts, pretrained = None, samples = 1000, batch_size = 8, epochs = 10, string_length = (2, 10), basepath = os.getcwd(), val_split = 0.2):
    import tensorflow as tf
    from edocr2 import keras_ocr
    basepath = os.path.join(basepath,
    f'detector_{time.gmtime(time.time()).tm_hour}'+f'_{time.gmtime(time.time()).tm_min}')

    text_generator = get_balanced_text_generator(alphabet, string_length)
    height, width = 640, 640
    backgrounds = get_backgrounds(height, width, samples)

    image_gen_params = {
    'height': height,
    'width': width,
    'text_generator': text_generator,
    'font_groups': {alphabet: fonts},  # Use all fonts
    'font_size': (20, 80),
    'margin': 25,
    'rotationZ': (-90, 120)
    }

    # Create image generators for training and validation
    image_generator_train  = generate_drawing_imgs(image_gen_params, backgrounds)
    image_generator_val  = generate_drawing_imgs(image_gen_params, backgrounds)

    detector = keras_ocr.detection.Detector(weights='clovaai_general')
    if pretrained:
        detector.model.load_weights(pretrained)
    
    detection_train_generator = detector.get_batch_generator(image_generator=image_generator_train,batch_size=batch_size)
    detection_val_generator = detector.get_batch_generator(image_generator=image_generator_val,batch_size=batch_size)

    detector.model.fit(
        detection_train_generator,
        steps_per_epoch=math.ceil((1 - val_split) * samples / batch_size),
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=5),
            tf.keras.callbacks.CSVLogger(f'{basepath}.csv'),
            tf.keras.callbacks.ModelCheckpoint(filepath=f'{basepath}.keras')
        ],
        validation_data=detection_val_generator,
        validation_steps=math.ceil(val_split * samples / batch_size),
        batch_size=batch_size
    )
    return basepath

############ Standard Training ################################################
def train_detector(data_path, batch_size = 8, epochs = 10, val_split = 0.2, pretrained = None):
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    import os
    import time
    import math

    # Set basepath for saving logs and model
    basepath = os.path.join(basepath, f'detector_{time.gmtime(time.time()).tm_hour}_{time.gmtime(time.time()).tm_min}')

############ Testing ##########################################################
def test_recog(test_path, recognizer):

    # To track ground truth and predictions for word-level accuracy
    total_chars = 0  # Total number of characters in all labels
    correct_chars = 0  # Total number of correctly predicted characters

    samples = len(os.listdir(test_path)) / 2
    
    for i in range(1, int(samples) + 1):
        img = cv2.imread(os.path.join(test_path, f"{i}.png"))
        with open(os.path.join(test_path, f"{i}.txt"), 'r') as txt_file:
            label = txt_file.read().strip()
        pred = recognizer.recognize(image = img)
        print(f'ground truth: {label} | prediction: {pred}')

        correct_in_sample = sum(1 for x, y in zip(label, pred) if x == y)
        correct_chars += correct_in_sample
        total_chars += len(label)

        sample_char_accuracy = (correct_in_sample / len(label)) * 100 if len(label) > 0 else 0
        print(f"Sample character accuracy: {sample_char_accuracy:.2f}%")

    # Calculate and print overall character-level accuracy
    overall_char_accuracy = (correct_chars / total_chars) * 100 if total_chars > 0 else 0
    print(f"\nTotal Samples: {samples}")
    print(f"Total characters: {total_chars}")
    print(f"Correctly predicted characters: {correct_chars}")
    print(f"Overall character-level accuracy: {overall_char_accuracy:.2f}%")

def test_detect(test_path, detector, show_img = False):

    samples = len(os.listdir(test_path)) / 2
    iou_scores =[]
    
    for i in range(1, int(samples) + 1):
        img = cv2.imread(os.path.join(test_path, f"img_{i}.png"))
        gt = []

        with open(os.path.join(test_path, f"gt_img_{i}.txt"), 'r') as txt_file:
            for line in txt_file:
                # Split the line by commas and strip any whitespace
                parts = line.strip().split(',')
                
                # Extract the coordinates (first 8 values) and the character (last value)
                coords = np.array([(int(parts[0]), int(parts[1])),
                                (int(parts[2]), int(parts[3])),
                                (int(parts[4]), int(parts[5])),
                                (int(parts[6]), int(parts[7]))])
                
                # Append a tuple of (coords, char) to the result list
                gt.append(coords)

        pred = detector.detect([img])

         # Calculate IoU for each predicted box with the closest ground truth box
        for pred_box in pred[0]:
            best_iou = 0.0
            for gt_box in gt:
                iou = calculate_iou(pred_box, gt_box)
                best_iou = max(best_iou, iou)  # Track the best IoU score for this prediction

            iou_scores.append(best_iou)

        if show_img:
            for box in pred:
                for xy in box:
                    pts=np.array([(xy[0]),(xy[1]),(xy[2]),(xy[3])], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

            for xy in gt:
                pts=np.array([(xy[0]),(xy[1]),(xy[2]),(xy[3])], dtype=np.int32)
                cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)   

            cv2.imshow('Image with Oriented Bounding Box', img)
            cv2.waitKey(0)  # Wait for a key press to close the image
            cv2.destroyAllWindows()
    
    # Print the average IoU score
    if iou_scores:
        print(f"Average IoU: {np.mean(iou_scores)}")
    else:
        print("No predictions found.")

def calculate_iou(predicted_polygon, ground_truth_polygon):
    """
    Calculate IoU (Intersection over Union) between two polygons.
    """
    from shapely.geometry import Polygon
    pred_poly = Polygon(predicted_polygon)
    gt_poly = Polygon(ground_truth_polygon)

    if not pred_poly.is_valid or not gt_poly.is_valid:
        return 0.0

    # Calculate intersection and union areas
    intersection_area = pred_poly.intersection(gt_poly).area
    union_area = pred_poly.union(gt_poly).area

    if union_area == 0:
        return 0.0

    iou = intersection_area / union_area
    return iou

#TODO: train_detect function on given dataset