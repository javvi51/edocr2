# Detailed installing instructions
This instalation guide for edocr includes steps to install WSL, Anaconda, Cuda and VS Code. If you have experience with this, jump to step 4.

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
   
2. Install **CUDA 11.8** from the [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).

3. Add CUDA to your `.bashrc` file:
   - Follow the instructions in this [Ask Ubuntu post](https://askubuntu.com/questions/885610/nvcc-version-command-says-nvcc-is-not-installed) to configure your environment.



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
1. Create your conda environment (Python 3.11)
   ```bash
   conda create -n edocr2 python=3.11 -y
   conda activate edocr2
   ```

2. Install **TensorFlow** with CUDA support:
   ```bash
   pip install tensorflow[and-cuda]
   ```
3. Install edocr requirements:
   ```bash
   pip install -r 'requirements.txt'
   ```

4. Install **Tesseract OCR** and **pytesseract**:
   - Install the necessary dependencies:
     ```bash
     sudo apt-get install libleptonica-dev tesseract-ocr libtesseract-dev python3-pil tesseract-ocr-eng tesseract-ocr-script-latn tesseract-ocr-nor
     ```
   - For swedish:
     ```bash
     sudo apt-get install tesseract-ocr-swe
     ```
   - Install Tesseract and pytesseract:
     ```bash
     pip install tesseract
     pip install pytesseract
     ```