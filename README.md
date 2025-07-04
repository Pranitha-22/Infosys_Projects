# 🧠 Infosys Deep Learning Projects

This repository contains two advanced deep learning projects completed during Infosys training. It includes Conditional GANs for class-wise image generation and object classification for memory-efficient CCTV surveillance.

[![Open in Colab]:https://colab.research.google.com/drive/1E8oS8nwO03yHOr_bH5RKhiNvKHc__rhd?usp=sharing]

---

## 🎨 Project 1: Conditional GAN using CIFAR-10

### 📌 Objective
Train a Conditional Generative Adversarial Network (CGAN) on the CIFAR-10 dataset to generate fake images conditioned on user-specified class labels.

### 📊 Dataset: CIFAR-10
- 60,000 RGB images (32x32 pixels)
- 10 classes: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

### 🧠 Highlights
- Generator and Discriminator implemented using Keras Functional API
- Label embedding to guide generation process
- User input for image class
- Generates realistic fake images

---

## 🎥 Project 2: Object Classification for Automated CCTV

### 📌 Objective
Design a CNN-based image classification system to reduce unnecessary CCTV recording by detecting objects (humans, animals, vehicles) and triggering recording only when needed.

### 📊 Dataset
- Fashion-MNIST: 60,000 grayscale images (28x28)
- CIFAR-10: For concept extension and future deployment

### 🧠 Highlights
- Memory-efficient design
- Uses deep learning to identify relevant objects
- Hyperparameter tuning for model improvement
- Evaluated with standard benchmarks

---

## 💻 Technologies Used

- Python
- Google Colab
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📦 Requirements

Below are the Python packages required to run the notebook:

tensorflow>=2.10.0
numpy
pandas
matplotlib
seaborn
scikit-learn

✅ You can install them with:
!pip install tensorflow numpy pandas matplotlib seaborn scikit-learn

🚀 How to Run
Clone the repository:

```bash
git clone https://github.com/your-username/infosys-deep-learning-projects.git
cd infosys-deep-learning-projects
Open the notebook using Google Colab or Jupyter.

Run the cells in Infosys_Projects.ipynb.

You can also open the notebook directly in Colab

```

