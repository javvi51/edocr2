import random, time, os, math, cv2 
import numpy as np

def get_text_generator(alphabet, string_length = (5, 10), lowercase=False):
    '''
    Generates a sentence.
    Args:
        alphabet: string of characters
        lowercase: convert alphabet to lowercase
        max_string_length: maximum number of characters in the sentence
    Return:
        sentence string
    '''
    while True:
        sentence = random.choices(alphabet, k=random.randint(string_length[0], string_length[1]))
        sentence = "".join(sentence)
        if lowercase:
            sentence = sentence.lower()
        yield sentence

def get_backgrounds(height, width, samples):
    def draw_random_lines_and_arcs(image):
        """Draw random arcs, horizontal and vertical lines on the image.
        
        Args:
        image: The image to draw on.
        """
        height, width, _ = image.shape
        
        # Number of lines and arcs to draw
        num_lines = random.randint(1, 5)  # Adjust as needed
        num_arcs = 0#random.randint(1, 3)   # Adjust as needed
        
        # Draw random horizontal and vertical lines
        for _ in range(num_lines):
            start_x = random.randint(0, width - 1)
            start_y = random.randint(0, height - 1)
            end_x = random.randint(0, width - 1)
            end_y = random.randint(0, height - 1)
            gray_value = random.randint(100, 255)  # Random grayscale color
            color = (gray_value, gray_value, gray_value)
            thickness = random.randint(1, 10)  # Random stroke size
            cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)
        
        # Draw random arcs
        for _ in range(num_arcs):
            center_x = random.randint(0, width - 1)
            center_y = random.randint(0, height - 1)
            axes = (random.randint(10, width // 2), random.randint(10, height // 2))
            angle = random.randint(0, 360)
            start_angle = random.randint(0, 30)
            end_angle = start_angle + random.randint(30, 360 - start_angle)
            gray_value = random.randint(100, 255)  # Random grayscale color
            color = (gray_value, gray_value, gray_value)
            thickness = random.randint(1, 10)  # Random stroke size
            cv2.ellipse(image, (center_x, center_y), axes, angle, start_angle, end_angle, color, thickness)


    backgrounds =[]
    for _ in range(samples):
        backg = np.zeros((height, width, 3), dtype="uint8")
        draw_random_lines_and_arcs(backg)
        backgrounds.append(backg)

    return backgrounds

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

    text_generator = get_text_generator(alphabet, string_length)

    image_gen_params = {
    'height': 640,
    'width': 640,
    'text_generator': text_generator,
    'font_groups': {alphabet: fonts},  # Use all fonts
    'font_size': (20, 120),
    'margin': 10
    }

    # Create image generators for training and validation
    image_generators_train = keras_ocr.data_generation.get_image_generator(**image_gen_params)
    image_generators_val = keras_ocr.data_generation.get_image_generator(**image_gen_params)
    
    recognizer = keras_ocr.recognition.Recognizer(alphabet=alphabet)
    if pretrained:
        recognizer.model.load_weights(pretrained)
    recognizer.compile()
    for layer in recognizer.backbone.layers:
        layer.trainable = False

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

def train_synth_detector(alphabet, fonts, pretrained = None, samples = 1000, batch_size = 8, epochs = 10, string_length = (2, 10), basepath = os.getcwd(), val_split = 0.2):
    import tensorflow as tf
    from edocr2 import keras_ocr
    basepath = os.path.join(basepath,
    f'detector_{time.gmtime(time.time()).tm_hour}'+f'_{time.gmtime(time.time()).tm_min}')

    text_generator = get_text_generator(alphabet, string_length)
    height, width = 640, 640
    backgrounds = get_backgrounds(height, width, samples)

    image_gen_params = {
    'height': height,
    'width': width,
    'text_generator': text_generator,
    'backgrounds': backgrounds,
    'font_groups': {alphabet: fonts},  # Use all fonts
    'font_size': (20, 120),
    'margin': 20,
    'rotationZ': (-30, 30)
    }

    # Create image generators for training and validation
    image_generator_train = keras_ocr.data_generation.get_image_generator(**image_gen_params)
    image_generator_val = keras_ocr.data_generation.get_image_generator(**image_gen_params)

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
            yield image, text
        '''else:
            print(f"Skipping sample due to low white pixel ratio ({white_pixel_ratio:.2%})")'''

def save_recog_samples(alphabet, fonts, sample_count, recognizer, save_path = './recog_samples'):
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
    label_filename = os.path.join(save_path, f'labels.txt')
    # Generate and save the samples
    for i in range(sample_count):

        text_generator = get_text_generator(alphabet, (5, 10))

        image_gen_params = {
        'height': 640,
        'width': 640,
        'text_generator': text_generator,
        'font_groups': {alphabet: fonts},  # Use all fonts
        'font_size': (20, 120),
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
        image_filename = os.path.join(save_path, f'sample_{i + 1}.png')
        cv2.imwrite(image_filename, image)
        
        # Save the label in a text file
        
        with open(label_filename, 'a') as label_file:
            label_file.write(text + '\n')

def save_detect_samples(alphabet, fonts, sample_count, save_path = './detect_samples'):
    from edocr2.keras_ocr import data_generation
    os.makedirs(save_path, exist_ok=True)
    label_filename = os.path.join(save_path, f'labels.txt')

    text_generator = get_text_generator(alphabet, (5, 10))
    height, width = 640, 640
    backgrounds = get_backgrounds(height, width, sample_count)

    image_gen_params = {
    'height': height,
    'width': width,
    'text_generator': text_generator,
    'backgrounds': backgrounds,
    'font_groups': {alphabet: fonts},  # Use all fonts
    'font_size': (20, 120),
    'margin': 20,
    'rotationZ': (-90, 90)
    }

    for i in range(sample_count):
        # Create image generators for training and validation
        image_generator_train = data_generation.get_image_generator(**image_gen_params)
        image, lines = next(image_generator_train)
        # Save the image
        image_filename = os.path.join(save_path, f'sample_{i + 1}.png')
        cv2.imwrite(image_filename, image)
        
        # Save the label in a text file
        with open(label_filename, 'a') as label_file:
            for l in lines[0]:
                array = l[0]
                label_file.write('\n'.join(' '.join(map(str, row)) for row in array) + '\n')
            label_file.write('\n\n')

def train_recog():
    pass

def train_detector():
    pass

#TODO: Drawing patches on background generator, fix font size
#TODO: train_recog function on given dataset (+get_alfa modificator) Tolerances???
#TODO: train_detect function on given dataset
#TODO: get standard fonts and filter them