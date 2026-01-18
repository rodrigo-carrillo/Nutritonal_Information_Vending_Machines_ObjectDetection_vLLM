# Vending Machines Nutrition

## Directory structure

## Python Scripts

| Script | Explanation |
|------------------------------------|------------------------------------|
| From VOC to YOLO.ipynb | Transforms data in VOC format to TXT format needed for YOLO training. HoloSelecta was in VOC format originally. The main function also reduces the number of total classes in the original dataset to a single class; we were not interested in identifying each product, but in detecting all objects in the vending machine. |
| YOLO_Fine_Tune.ipynb & YOLO_Fine_Tune.py | Fine-tuning of YOLO v11 model for object detection using the HoloSelecta dataset. Both scripts contain largely the same code, though the .py file was used with *caffeinate* in MacOS so that the system will not go to sleep. The fine-tuning was conducted for one single label (object) rather than all original labels available in HoloSelecta. The fine-tuned models are stored in the 02_Models folder, one for each iteration with different Optimizers: SGD, Adam and AdamW. |
| Segmentation.ipynb | This applies the fine-tuned YOLO model to crop each individual product out of the vending machine full photograph. The copped images (one small image per object/food product in the vending machine) are in the 03_Cropped_Objects folder (the file name of the cropped images contains the probability of the object). |
| VLLMs.ipynb & VLLMs.py | Reads the cropped images (each individual food product), passes it to the a vLLM to get information out of the image (e.g., brand name). Zero-shot prompt. The output is a .CVS and .JSON file with the information extracted by the vLLM. The outputs are stored in 04_Output_Data (CSV and JSON). |

## Preliminary results

Manual verification of some of the cropped images.
[Sample_Results.tiff](https://github.com/user-attachments/files/24692756/Sample_Results.tiff)
