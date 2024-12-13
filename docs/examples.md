# Examples and How to Use the Tool

There are four test examples that provide code to help you start running inference on your own engineering drawings. Depending on the capabilities you want to use, these files offer:

- **`test_drawing.py`**: Tests a single drawing and shows the segmentation and detection results for eDOCr2.
- **`test_all.py`**: Benchmarks using the available drawings in the `/tests` folder with eDOCr2.
- **`test_llm.py`**: Provides three modalities:  
  1. Segmentation + Qwen2-VL (or any other vision-language model from Hugging Face you can run locally).  
  2. Segmentation + GPT-4o.  
  3. Raw GPT-4o inference.  
- **`test_train.py`**: For training a new synthetic detector or recognizer.

### Model Setup
To use the recognizers in these files, download the models and place them in a folder named `models` inside the `edocr2` directory. The models are available in the [Releases](https://github.com/javvi51/edocr2/releases).

---

## `test_all.py`

This file runs a loop based on the `test_drawing.py` code over all test images in the `/tests` folder and computes metrics based on the results. 

It provides two main functions:
1. **Ground Truth Reader**: Reads the ground truth data from JSON files.
2. **Metrics Computation**: Computes metrics for each drawing.

Finally, global micro and macro average metrics are calculated and printed in the terminal.

---

## `test_llm.py`

This script processes all images in a folder using LLM integration. It supports three modes, controlled by setting the following flags to `True` or `False`:

```python
qwen, edocr_gpt, raw_gpt = False, True, False
# This configuration runs eDOCr2 segmentation plus GPT-4o OCR. 
```
### Requirements

To run this script, create a .env file in the root directory and specify your OpenAI API key and Hugging Face token as follows:
```python
HF_TOKEN = ########################
OPENAI_API_KEY = ########################
```

## test_train.py
This script runs end-to-end training and testing for both the detector and recognizer. It is split into two sections, if you wish to train only the detector or the recognizer, remove the code for the other.

The training process handles both synthetic data generation and model evaluation.
