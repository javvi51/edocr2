import random, time, os, math, cv2
import numpy as np
from collections import Counter
import typing

############ Synthetic Generation ###############################################
def get_and_process_fonts(dir_target):

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
                        move_file_to_directory(font_path, dir_target)
    shutil.rmtree('fonts')

def check_fonts(folder_path = 'edocr2/tools/dimension_fonts/', characters = '(),.+-±:/°"⌀'):
    from PIL import Image, ImageDraw, ImageFont
    def draw_character_cv2(char, font_path, font_size, img_width, img_height):
        # Create a blank image using PIL (RGBA mode to handle transparency)
        pil_image = Image.new('RGBA', (img_width, img_height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(pil_image)

        # Load the TTF font
        font = ImageFont.truetype(font_path, font_size)

        # Get the size of the text to center it in the image
        bbox = font.getbbox(char)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate the position to center the character
        position = ((img_width - text_width) // 2, (img_height - text_height) // 2)

        # Draw the character onto the PIL image
        draw.text(position, char, font=font, fill=(0, 0, 0, 255))

        # Convert the PIL image to a format OpenCV can work with (BGR mode)
        cv_image = np.array(pil_image)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGRA)  # Preserve transparency

        return cv_image

    files = os.listdir(folder_path)
    for i in files:
        font_path = os.path.join(folder_path, i)
        img = draw_character_cv2(characters, font_path, 50, 400, 400)

        # Display the result with OpenCV
        cv2.imshow('Character', img)
        key = cv2.waitKey(0)

        if key == ord('1'):
            os.remove(font_path)
            print(f"File {i} has been removed.")
        elif key == ord('0'):
            print(f"File {i} was not removed.")

        cv2.destroyAllWindows()

def get_balanced_text_generator(alphabet, string_length=(5, 10), lowercase=False, bias_chars = '', bias_factor = 0.3):
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
        weights = {}
        for char in alphabet:
            # Apply the bias factor for specified characters
            weight = total_generated - symbol_counter[char] + 1
            if char in bias_chars:
                weight += bias_factor
            weights[char] = weight
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
            text_mask = text_img < 127
            # Identify where the background has black pixels (0 value)
            background_black_mask = background_gray < 127
            # Check if any black background pixels overlap with the white text pixels
            overlap = np.any(np.logical_and(text_mask, background_black_mask))

            return overlap

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
    
    def reposition(text_img, lines):
        new_lines = []
        for line in lines:
            x_coords = []
            y_coords = []
            for li in line:
                x_coords.extend([li[0][0][0], li[0][1][0], li[0][2][0], li[0][3][0]])  # [x1, x2, x3, x4]
                y_coords.extend([li[0][0][1], li[0][1][1], li[0][2][1], li[0][3][1]])  # [y1, y2, y3, y4]
            
            x_min = int(min(x_coords))
            y_min = int(min(y_coords))
            x_max = int(max(x_coords))
            y_max = int(max(y_coords))
            
            # Crop the text region using the bounding box coordinates
            cropped_text = text_img[y_min:y_max, x_min:x_max]
            x_offset = random.randint(10, text_img.shape[1] - x_max + x_min - 10)
            y_offset = random.randint(10, text_img.shape[0] - y_max + y_min - 10)
            
            text_img[y_offset:y_offset+ cropped_text.shape[0], x_offset:x_offset+ cropped_text.shape[1]] = cropped_text
            text_img[y_min:y_max, x_min:x_max] = 0
            new_line = []
            for li in line:
                new_li = []
                for coord in li[0]:  # Iterate through each (x, y) pair in the bounding box
                    new_x = coord[0] - x_min + x_offset
                    new_y = coord[1] - y_min + y_offset
                    new_li.append([new_x, new_y])
                new_line.append([new_li, li[1]])
            new_lines.append(new_line)

        return text_img, new_lines
    
    from edocr2.keras_ocr import data_generation

    while True:
        backg = random.choice(backgrounds)
        # Initialize the final image as the background
        image = backg.copy()
        lines = []  # Store the bounding boxes for all text images

        # Randomly choose a number of text images to place (between 1 and 5, for example)
        num_images = random.randint(1, 5)

        for _ in range(num_images):
            for _ in range(100):  # Retry mechanism if overlap occurs
                image_gen = data_generation.get_image_generator(**image_gen_params)
                text_img, new_lines = next(image_gen)
                text_img, new_lines = reposition(text_img, new_lines)
                _, binary_text_img = cv2.threshold(cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY_INV)
                # Check if the new text image overlaps with the current image
                
                if not check_overlap(binary_text_img, image):
                    # If no overlap, apply the text image onto the background
                    image = apply_text_on_background(text_img, binary_text_img, image)

                    # Compact the bounding boxes and add them to the list
                    new_lines = compact_bounding_box(new_lines)
                    lines.extend(new_lines)
                    break  # Exit the loop once the image has been successfully placed
            else:
                continue  # Retry if overlap occurred

        # Yield the final image with the applied text and the compacted bounding boxes
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

    text_generator = get_balanced_text_generator(alphabet, (1, 10))
    height, width = 640, 640
    backgrounds = get_backgrounds(height, width, samples)

    image_gen_params = {
    'height': height,
    'width': width,
    'text_generator': text_generator,
    'font_groups': {alphabet: fonts},  # Use all fonts
    'font_size': (25, 50),
    'margin': 20,
    'rotationZ': (-90, 90)
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

def train_synth_recognizer(alphabet, fonts, pretrained = None, bias_char = '', samples = 1000, batch_size = 256, epochs = 10, string_length = (5, 10), basepath = os.getcwd(), val_split = 0.2):
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
    current_time = time.localtime(time.time())
    basepath = os.path.join(basepath,
    f'recognizer_{current_time.tm_hour}_{current_time.tm_min}')

    text_generator = get_balanced_text_generator(alphabet, string_length, bias_chars=bias_char)

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
    with open(f'{basepath}.txt', 'w') as file:
        file.write(alphabet)
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

def train_synth_detector(alphabet, fonts, pretrained = None, samples = 100, batch_size = 8, epochs = 1, string_length = (1, 10), basepath = os.getcwd(), val_split = 0.2):
    import tensorflow as tf
    from edocr2 import keras_ocr
    current_time = time.localtime(time.time())
    basepath = os.path.join(basepath,
    f'detector_{current_time.tm_hour}_{current_time.tm_min}')

    text_generator = get_balanced_text_generator(alphabet, string_length)
    height, width = 640, 640
    backgrounds = get_backgrounds(height, width, samples)

    image_gen_params = {
    'height': height,
    'width': width,
    'text_generator': text_generator,
    'font_groups': {alphabet: fonts},  # Use all fonts
    'font_size': (25, 50),
    'margin': 0,
    'rotationZ': (-90, 90)
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

############ Testing ##########################################################
def compare_characters(label, prediction):
    # Count occurrences of each character in label and prediction
    label_chars = Counter(label)    # e.g., {'1': 1, '4': 1, '0': 1}
    pred_chars = Counter(prediction)  # e.g., {'4': 1, '0': 1}

    correct_count = 0

    # Iterate over characters in the prediction
    for char in pred_chars:
        if char in label_chars:
            # Add the minimum of occurrences in both to correct_count
            correct_count += min(pred_chars[char], label_chars[char])
    return correct_count

def get_cer(
    preds: typing.Union[str, typing.List[str]],
    target: typing.Union[str, typing.List[str]],
    ) -> float:
    
    def edit_distance(prediction_tokens: typing.List[str], reference_tokens: typing.List[str]) -> int:
        """ Standard dynamic programming algorithm to compute the Levenshtein Edit Distance Algorithm

        Args:
            prediction_tokens: A tokenized predicted sentence
            reference_tokens: A tokenized reference sentence
        Returns:
            Edit distance between the predicted sentence and the reference sentence
        """
        # Initialize a matrix to store the edit distances
        dp = [[0] * (len(reference_tokens) + 1) for _ in range(len(prediction_tokens) + 1)]

        # Fill the first row and column with the number of insertions needed
        for i in range(len(prediction_tokens) + 1):
            dp[i][0] = i
        
        for j in range(len(reference_tokens) + 1):
            dp[0][j] = j

        # Iterate through the prediction and reference tokens
        for i, p_tok in enumerate(prediction_tokens):
            for j, r_tok in enumerate(reference_tokens):
                # If the tokens are the same, the edit distance is the same as the previous entry
                if p_tok == r_tok:
                    dp[i+1][j+1] = dp[i][j]
                # If the tokens are different, the edit distance is the minimum of the previous entries plus 1
                else:
                    dp[i+1][j+1] = min(dp[i][j+1], dp[i+1][j], dp[i][j]) + 1

        # Return the final entry in the matrix as the edit distance     
        return dp[-1][-1]
    """ Update the cer score with the current set of references and predictions.

    Args:
        preds (typing.Union[str, typing.List[str]]): list of predicted sentences
        target (typing.Union[str, typing.List[str]]): list of target words

    Returns:
        Character error rate score
    """
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(target, str):
        target = [target]

    total, errors = 0, 0
    for pred_tokens, tgt_tokens in zip(preds, target):
        errors += edit_distance(list(pred_tokens), list(tgt_tokens))
        total += len(tgt_tokens)

    if total == 0:
        return 0.0

    cer = errors / total

    return cer

def test_recog(test_path, recognizer):

    # To track ground truth and predictions for word-level accuracy
    total_chars = 0  # Total number of characters in all labels
    pred_chars = 0 
    cer = []
    correct_chars = 0  # Total number of correctly predicted characters
    samples = len(os.listdir(test_path)) / 2
    
    for i in range(1, int(samples) + 1):
        img = cv2.imread(os.path.join(test_path, f"{i}.png"))
        with open(os.path.join(test_path, f"{i}.txt"), 'r') as txt_file:
            label = txt_file.read().strip()
        pred = recognizer.recognize(image = img)
        print(f'ground truth: {label} | prediction: {pred}')

        correct_in_sample = compare_characters(label, pred)
        correct_chars += correct_in_sample
        total_chars += len(label)

        sample_char_recall = (correct_in_sample / len(label)) * 100 if len(label) > 0 else 0
        sample_cer = get_cer(pred, label) * 100
        cer.append(sample_cer)
        pred_chars += len(pred)
        print(f"Sample character Recall: {sample_char_recall:.2f}%")
        print(f"Sample character CER: {sample_cer:.2f}%")

    # Calculate and print overall character-level accuracy
    overall_char_recall = (correct_chars / pred_chars) * 100 if pred_chars > 0 else 0
    overall_cer = np.mean(cer)

    print(f"Character Recall: {overall_char_recall:.2f}%")
    print(f"CER: {overall_cer:.2f}%")

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

