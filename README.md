# eDOCr2

A tool for performing segmentation and OCR on engineering drawings, primarily focused on mechanical or production drawings.

## Installation

Detailed installation steps can be found [here](https://github.com/javvi51/edocr2/blob/main/docs/install.md).

## How to Use

For quick testing, run the `test_drawing.py` file after downloading the recognizer models from [Releases](https://github.com/javvi51/edocr2/releases).

Other files are provided for additional functionality:
- **`test_train.py`**: For training custom synthetic recognizers or detectors.
- **`test_all.py`**: For benchmarking against all drawings available in the `/tests` folder.
- **`test_llm.py`**: For integration with language models (LLMs).

For more detailed information about using these files, refer to the [Examples](https://github.com/javvi51/edocr2/blob/main/docs/examples.md) documentation.

## Citation

If you find this project useful in your research, please consider citing our paper (preprint):  
[http://dx.doi.org/10.2139/ssrn.5045921](http://dx.doi.org/10.2139/ssrn.5045921)