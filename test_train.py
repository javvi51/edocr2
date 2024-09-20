import string, os
from edocr2.tools import train_tools


#region ############# Alphabet definition #################
GDT_symbols = '⏤⏥○⌭⌒⌓⏊∠⫽⌯⌖◎↗⌰'
FCF_symbols = 'ⒺⒻⓁⓂⓅⓈⓉⓊ'
Extra = '(),.+-±:/°"⌀'

alphabet_gdts = string.digits + ',.⌀ABCD' + GDT_symbols + FCF_symbols
alphabet_dimensions = string.digits + 'AaBCDRGHhMmnx' + Extra
#endregion

gdt_fonts=[]
for i in os.listdir('edocr2/tools/gdt_fonts'):
    gdt_fonts.append(os.path.join('edocr2/tools/gdt_fonts', i))


#region ############## Detector ##############################

########## Training Detector ###############################
detect_basepath = train_tools.train_synth_detector(alphabet_dimensions, gdt_fonts, samples = 100, epochs =5, batch_size=8)


######### Testing Detector #################################
#train_tools.save_detect_samples(alphabet_dimensions, gdt_fonts, 10)
'''from edocr2.keras_ocr.detection import Detector
detect_basepath = None #'detector_14_53.keras'
detector = Detector()
detector.model.load_weights(detect_basepath + '.keras')
train_tools.test_detect('detect_samples', detector)'''

#endregion

#region ############## Recognizer ############################

'''########## Training Recognizer #############################
recog_basepath = train_tools.train_synth_recognizer(alphabet_gdts, gdt_fonts, samples = 35000, epochs = 5, batch_size=256)

########## Testing Recognizer ##############################

from edocr2.keras_ocr.recognition import Recognizer
recognizer = Recognizer(alphabet=alphabet_gdts)
train_tools.save_recog_samples(alphabet_gdts, gdt_fonts, 30, recognizer)
#recog_basepath = 'recognizer_8_32.keras'
recognizer.model.load_weights(recog_basepath + '.keras')
train_tools.test_recog('recog_samples', recognizer)'''

#endregion