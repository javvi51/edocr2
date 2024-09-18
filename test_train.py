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

#train_tools.train_synth_detector(alphabet_dimensions, gdt_fonts, samples = 100, epochs =10, batch_size=8)
train_tools.train_synth_recognizer(alphabet_gdts, gdt_fonts, samples = 100, epochs = 30, batch_size=256)
