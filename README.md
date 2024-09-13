# edocr2

So far, installation steps:
-In powershell run -> wsl --install
-Start up/Turn Windows features on off -> check Virtual Machine and Windows Subsystem for Linux
In Ubuntu:
-Install Anaconda by checking latest from https://repo.anaconda.com/archive/ and run -> wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh 
-Install Cuda 12.3 from Nvidia cuda toolkit
-Add them to .bashrc following https://askubuntu.com/questions/885610/nvcc-version-command-says-nvcc-is-not-installed
-Install Tensorflow -> pip install tensorflow[and-cuda]
-Install VS code -> code .
-Clone repo
-Install cv2 -> pip install opencv-python
-Install pdf2image -> pip install pdf2image
-Install pytesserat:
    -sudo apt-get install libleptonica-dev tesseract-ocr libtesseract-dev python3-pil tesseract-ocr-eng tesseract-ocr-script-latn
    -pip install tesseract
    -pip install pytesseract

