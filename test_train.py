import string, os
from edocr2.tools import train_tools

#region ############# Alphabet and fonts definition #################
GDT_symbols = '⏤⏥○⌭⌒⌓⏊∠⫽⌯⌖◎↗⌰'
FCF_symbols = 'ⒺⒻⓁⓂⓅⓈⓉⓊ'
Extra = '(),.+-±:/°"⌀='

alphabet_gdts = string.digits + ',.⌀ABCD' + GDT_symbols + FCF_symbols
alphabet_dimensions = string.digits + 'AaBCDRGHhMmnxZtd' + Extra

gdt_fonts=[]
for i in os.listdir('edocr2/tools/gdt_fonts'):
    gdt_fonts.append(os.path.join('edocr2/tools/gdt_fonts', i))


dimension_fonts = []
for i in os.listdir('edocr2/tools/dimension_fonts'):
    dimension_fonts.append(os.path.join('edocr2/tools/dimension_fonts', i))
#endregion

#region ############## Detector ##############################

########## Training Detector ###############################
#detect_basepath = train_tools.train_synth_detector(alphabet_dimensions, dimension_fonts, samples = 200, epochs =1, batch_size=8, basepath = 'edocr2/models')


######### Testing Detector #################################
#train_tools.save_detect_samples(alphabet_dimensions, dimension_fonts, 2)
'''from edocr2.keras_ocr.detection import Detector
#detect_basepath = 'edocr2/models/detector_8_58'
detector = Detector()
detector.model.load_weights(detect_basepath + '.keras')
train_tools.test_detect('detect_samples', detector, show_img=True)'''

#endregion

#region ############## Recognizer ############################

########## Training Recognizer #############################
recog_basepath = train_tools.train_synth_recognizer(alphabet_dimensions, dimension_fonts, bias_char=Extra, samples = 20000, epochs = 5, batch_size=256, basepath = 'edocr2/models')

########## Testing Recognizer ##############################

from edocr2.keras_ocr.recognition import Recognizer
recognizer = Recognizer(alphabet=alphabet_dimensions)
train_tools.save_recog_samples(alphabet_dimensions, dimension_fonts, 30, recognizer)
#recog_basepath = 'recognizer_8_32'
recognizer.model.load_weights(recog_basepath + '.keras')
train_tools.test_recog('recog_samples', recognizer)

#endregion