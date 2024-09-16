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

Follow these steps to set up the environment for the project.

## 1. Enable Windows Subsystem for Linux (WSL)

1. Open **PowerShell** as Administrator and run the following command to install WSL:
    ```bash
    wsl --install
    ```
2. Go to **Turn Windows features on or off** and ensure the following features are enabled:
   - **Virtual Machine Platform**
   - **Windows Subsystem for Linux**

## 2. Set Up Anaconda and Cuda in WSL

1. Open the Ubuntu terminal in WSL and install **Anaconda**:
   - Check the latest version of Anaconda from [Anaconda Archive](https://repo.anaconda.com/archive/).
   - Run the following command (replace with the latest version):
     ```bash
     wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
     ```
   
2. Install **CUDA 12.3** from the [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).

3. Add CUDA to your `.bashrc` file:
   - Follow the instructions in this [Ask Ubuntu post](https://askubuntu.com/questions/885610/nvcc-version-command-says-nvcc-is-not-installed) to configure your environment.

4. Create your conda environment (Python 3.9)
   ```bash
   conda create -n edocr python=3.9
   ```

4. Install **TensorFlow** with CUDA support:
   ```bash
   pip install tensorflow[and-cuda]
   ```

## 3. Install and Configure VS Code

1. Install **Visual Studio Code**:
   ```bash
   sudo apt install code
   ```
2. Open the project in VS Code:
   ```bash
   code .
   ```

## 4. Install Required Python Packages

1. Install edocr requirements:
   ```bash
   pip install -r 'requirements.txt'
   ```

2. Install **Tesseract OCR** and **pytesseract**:
   - Install the necessary dependencies:
     ```bash
     sudo apt-get install libleptonica-dev tesseract-ocr libtesseract-dev python3-pil tesseract-ocr-eng tesseract-ocr-script-latn
     ```
   - Install Tesseract and pytesseract:
     ```bash
     pip install tesseract
     pip install pytesseract
     ```