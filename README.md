# Towards a Safer Construction Environment: Evaluating a Simple CNN for Safety Classification
### By: Darius Vincent C. Ardales

This project explores workplace safety compliance in construction sites by evaluating a Convolutional Neural Network (CNN) for binary classification of images into "safe" (individuals wearing safety equipment) and "unsafe" (individuals not wearing safety equipment). It includes training, testing, and deploying the model for real-time classification through a FastAPI app, containerized with Docker, and hosted on Hugging Face Spaces.

## Prerequisites

Before you begin, ensure you have the following:

- **pip** (preferred if you also have `conda` for environment management)
- **Python 3.9.18** (strictly this version for compatibility)
- A virtual environment (optional but highly recommended)
- **Operating System**: This project was developed on a Microsoft Windows 11 laptop.

---

## Project Structure

Below is an organized overview of the project's folders, files, and their roles:

### **Folders**
- **`data`**
  - Contains two subfolders:
    - `safe`: Original images classified as "safe".
    - `unsafe`: Original images classified as "unsafe".
- **`augmented`**
  - Contains two subfolders:
    - `safe`: Augmented "safe" images specifically used for training.
    - `unsafe`: Augmented "unsafe" images specifically used for training.

### **Files**
- **`safety_classifier_model.ipynb`**
  - Jupyter notebook where all steps of the project are executed, including preprocessing, augmentation, model creation, training, testing, and analysis.
- **`safety_classifier.py`**
  - Executable FastAPI code for real-time binary image classification (safe/unsafe) using the trained SafetyCNN model.
- **`Dockerfile`**
  - Containerizes the FastAPI app for deployment on Hugging Face Spaces or other hosting platforms.
- **`safety_model.pth`**
  - Saved PyTorch model of the trained CNN for safety classification.
- **`requirements.txt`**
  - List of dependencies needed to run `safety_classifier.py`.

---

## Steps to Run the Project

Follow these steps to clone, set up, and run the project locally:

### 1. **Clone the Repository**
   ```bash
   git clone https://github.com/riu-rd/Worksite-Safety-Monitoring.git
   cd Worksite-Safety-Monitoring
   ```

### 2. Download the Model (Needed step because of GitHub memory constraints)
- Download safety_model.pth from the author's Hugging Face [HERE](https://huggingface.co/spaces/riu-rd/safety_classifier/blob/main/safety_model.pth).
- Place the file in the same directory as the project files.

### 3. Set Up a Virtual Environment
- If you don’t have a virtual environment, create one using `pip` or `conda`. Make sure the Python version is `3.9.18`.
- Example using `pip`:
  ```bash
  python -m venv env
  source env/bin/activate  # On Linux/MacOS
  env\Scripts\activate     # On Windows
   ```
- Example using `conda`:
  ```bash
  conda create --name safety_env python=3.9.18
  conda activate safety_env
   ```

### 4. Install Dependencies
  ```bash
  pip install -r requirements.txt
   ```

### 5. Run the FastAPI App Locally
  ```bash
  uvicorn safety_classifier:app --reload
   ```

---

## Deployment

- The FastAPI app is containerized using Docker and hosted live on Hugging Face Spaces. To view the project live, visit: [Live Link](https://riu-rd-safety-classifier.hf.space/)

---

## Notes and Recommendations
- Ensure that your environment matches the prerequisites for a smooth setup.
- For issues or contributions, please open an issue in the repository or submit a pull request.

---

# Happy Worksite Safety Classifying!