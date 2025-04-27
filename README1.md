### IMAGECLEF - MULTIMODAL VQA
## ABOUT THE TASK:
MultimodalReason is a new task focusing on Multilingual Visual Question Answering (VQA). The formulation of the task is the following:
Given an image of a question with 3-5 possible answers, participants must identify the single correct answer.
## ABOUT THE DATASET:
Training and dev/validation data split into 16,724 training and 4,208 dev/validation instances.
The data is provided in various languages as well : 
![image](https://github.com/user-attachments/assets/cc81dade-34f0-4410-95b1-f5990b38c9f8)
Each question will have questions as image/text , and details regarding what kind of question is given in the dataset itself. Task is to just
provide the correct answer for each question as an option label - each option may be images 
## ABOUT OUR METHODOLOGY:
We retrieved the dataset provided as parquet files (extracter.py) and parsed them to extract necessary data and store the retrieved dataset (Addpaths.py) , preprocessed the image and embedded / formatted it into a prompt to provide
our chosen vision language model - Qwen 2.5 VL 72b Instruct - which has been trained with multimodal capabilities (final.py) .
We retrieved the outputs and formatted them to the output submission format found in run.json .
