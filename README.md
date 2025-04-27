# ImageCLEF MultimodalReasoning

This project performs visual question answering (VQA) on the EXAMS-V dataset, a dataset of visual multiple-choice exam questions. The goal is to download the dataset, fetch associated images, and run a state-of-the-art vision-language model to predict answers to the exam questions.

## Project Overview

The EXAMS-V dataset contains exam questions with images, such as diagrams, graphs, or tables, along with multiple-choice answers. This project uses the Qwen2.5-VL-72B-Instruct model to analyze the visual content and answer the questions.

## Files and Their Purpose

- **Extracter.py**  
  Downloads the full 'test' split of the EXAMS-V dataset from Hugging Face in batches and saves it locally as `exams_v_test_full.json`.

- **AddPaths.py**  
  Reads the downloaded dataset JSON (`exams_v_test_full.json`), downloads all referenced images locally into a folder `downloaded_images/`, updates the dataset entries with local image paths, and saves the updated dataset as `exams_v_test_full_with_paths.json`.

- **final.py**  
  Loads the dataset with local image paths (`exams_v_test_full_with_paths.json`), runs the Qwen2.5-VL-72B-Instruct vision-language model to perform visual question answering, extracts predicted answer labels (A-E), and saves the results to `run.json`.

- **requirements.txt**  
  Lists all Python dependencies required to run the scripts.

## Setup Instructions

1. Clone or download this repository.

2. Create a Python virtual environment (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage Instructions

Run the scripts in the following order:

1. **Download the dataset:**

   ```bash
   python3 Extracter.py
   ```

   This will download the EXAMS-V test split and save it as `exams_v_test_full.json`.

2. **Download images and update dataset JSON:**

   ```bash
   python3 AddPaths.py
   ```

   This downloads all images referenced in the dataset and updates the JSON with local image paths, saving as `exams_v_test_full_with_paths.json`.

3. **Run visual question answering:**

   ```bash
   python3 final.py
   ```

   This runs the Qwen2.5-VL-72B-Instruct model on the dataset and saves the predicted answers to `run.json`.

## Output Files

- `exams_v_test_full.json`: The downloaded EXAMS-V test dataset.
- `exams_v_test_full_with_paths.json`: The dataset JSON updated with local image paths.
- `run.json`: The final predictions with answer keys for each question.

## Notes

- The model used in `final.py` requires a GPU with sufficient memory and CUDA support.
- The image downloads are saved in the `downloaded_images/` directory.
- The scripts include basic error handling and progress tracking.

## References

- [EXAMS-V Dataset on Hugging Face](https://huggingface.co/datasets/MBZUAI/EXAMS-V)
- [Qwen2.5-VL-72B-Instruct Model](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)
