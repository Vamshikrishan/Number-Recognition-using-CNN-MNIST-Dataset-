**📘 Number Recognition using CNN (MNIST Dataset)**

**📖 Overview**

This project demonstrates a **Deep Learning model (Convolutional Neural Network - CNN)** for recognizing handwritten digits (0–9) from the **MNIST dataset.**
The workflow includes:
-Importing & preprocessing the dataset
-Building and training a CNN model
-Saving & reusing the trained model
-Making predictions on handwritten digits
The project is built as a **Jupyter Notebook (number_recognition.ipynb)** and is designed to run seamlessly in **Google Colab.**

**📂 Project Structure**

DL_project/
  └── DL_project/
        ├── kaggle.json              # Kaggle API credentials (optional, if using Kaggle datasets)
        ├── number_recognition.ipynb # Main notebook (code, training, evaluation)
        └── final_mnist_cnn.h5       # Trained CNN model (saved weights)
    
**⚙️ Requirements**
You don’t need to install dependencies if using manually **Google Colab**, since it already has most libraries preinstalled.
If you want to run locally, install the following:
  pip install tensorflow keras matplotlib numpy pandas

**▶️ How to Run (Google Colab)**
1. Upload the project folder to your **Google Drive.**
   - Place the folder in your Drive root for easier access.
2. Open **Google Colab:**
   https://colab.research.google.com/
3. Mount Google Drive in Colab:
   from google.colab import drive
   drive.mount('/content/drive')
4. Navigate to the folder where you uploaded the project:
   %cd /content/drive/MyDrive/DL_project/DL_project
5. Open the notebook:
   - Run all cells in number_recognition.ipynb.
   - The notebook will:
     - Load dataset (MNIST)
     - Train the CNN
     - Save/load the model (final_mnist_cnn.h5)
     - Test predictions

**🧠 Model Details**
- **Model Type:** Convolutional Neural Network (CNN)
- **Dataset:** MNIST Handwritten Digits (28x28 grayscale images, 10 classes)
- **Architecture:**
  - Convolutional layers for feature extraction
  - MaxPooling for downsampling
  - Dense layers for classification
- **Output:** Digit prediction (0–9)

**📊 Results & Observations**
- Achieves high accuracy (~99%) on MNIST test set.
- Works well on clean, centered digits.
- May confuse similar digits (e.g., 4 vs 9, 5 vs 6).

**📑 File Explanations**
- number_recognition.ipynb → Main notebook to train/test the CNN.
- final_mnist_cnn.h5 → Pretrained model. Load this to skip training:
    from tensorflow.keras.models import load_model
    model = load_model("final_mnist_cnn.h5")
- kaggle.json → Kaggle credentials (use only if you’re downloading datasets from Kaggle).

**🚀 Future Improvements**
  - Extend model to recognize custom handwritten digits (outside MNIST).
  - Deploy as a simple web app (e.g., with Streamlit or Flask).
  - Experiment with deeper CNNs or regularization to further reduce misclassification.

**🙌 Credits**
- **Dataset:** http://yann.lecun.com/exdb/mnist/
- **Frameworks:** TensorFlow, Keras
