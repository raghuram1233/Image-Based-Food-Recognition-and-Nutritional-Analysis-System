# Computer Vision Project

This repository contains two main computer vision applications: **Object Dimension Measurement** and **Fruit Classification**. The project combines traditional computer vision techniques with modern deep learning approaches to solve real-world problems.

## 📁 Project Structure

```
Prof_Project/
├── main.py                 # Object dimension measurement script
├── Train.ipynb            # Fruit classification model training notebook
├── Test.ipynb             # Fruit classification testing notebook
├── Fruits_model.h5        # Pre-trained CNN model for fruit classification
├── requirements.txt       # Python dependencies
├── Data/                  # Dataset directory
│   ├── train/             # Training dataset
│   │   ├── Apple/         # Apple training images
│   │   ├── Banana/        # Banana training images
│   │   ├── Mango/         # Mango training images
│   │   ├── Orange/        # Orange training images
│   │   └── Pineapple/     # Pineapple training images
│   └── test/              # Testing dataset
│       ├── Apple/         # Apple test images
│       ├── Banana/        # Banana test images
│       ├── Mango/         # Mango test images
│       ├── Orange/        # Orange test images
│       └── Pineapple/     # Pineapple test images
├── imgs/                  # Sample images for testing
│   ├── apple.jpg          # Sample fruit image
│   └── test1.png          # Sample object measurement image
└── README.md              # Project documentation
```

## 🎯 Features

### 1. Object Dimension Measurement (`main.py`)

- Measures real-world dimensions of objects in images using computer vision
- Implements edge detection, contour analysis, and perspective correction
- Provides measurements in millimeters using a reference object for scale calibration
- Automatic object detection with bounding box calculation
- Real-time visualization of measurements overlaid on images

### 2. Fruit Classification Model Training (`Train.ipynb`)

- Complete CNN model training pipeline for fruit classification
- Data loading and preprocessing from organized dataset structure
- Model architecture design with Conv2D, MaxPooling, and Dropout layers
- Training with data augmentation using ImageDataGenerator
- Model evaluation and performance metrics
- Model saving in HDF5 format for deployment

### 3. Fruit Classification Testing (`Test.ipynb`)

- Load and test the pre-trained fruit classification model
- Real-time fruit classification with confidence scores
- Supports 5 fruit categories: Apple, Banana, Pineapple, Mango, Orange
- Image preprocessing and prediction pipeline
- Visualization of results with predictions

## 🛠️ Dependencies

### Required Libraries

The project uses the following Python libraries (see `requirements.txt` for specific versions):

```bash
# Core Machine Learning and Computer Vision
tensorflow>=2.8.0          # Deep learning framework
opencv-python>=4.5.0       # Computer vision library
numpy>=1.21.0              # Numerical computing

# Scientific Computing
scipy>=1.7.0               # Scientific computing utilities

# Computer Vision Utilities
imutils>=0.5.4             # Computer vision convenience functions

# Data Visualization
matplotlib>=3.5.0          # Plotting and visualization

# Jupyter Notebook Support
jupyter>=1.0.0             # Jupyter notebook environment
ipykernel>=6.0.0           # Jupyter kernel for Python
```

### Installation

Install all dependencies using pip:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install tensorflow opencv-python numpy scipy imutils matplotlib jupyter ipykernel
```

## 🚀 Usage

### Object Dimension Measurement

1. **Configure the script:**

   ```python
   image_path = 'imgs/test1.png'    # Path to your image
   ref_obj_width = 25               # Width of reference object in mm
   ```

2. **Run the measurement:**

   ```bash
   python main.py
   ```

3. **How it works:**
   - The script automatically detects objects in the image
   - Uses the first detected object as a reference for scale calibration
   - Calculates and displays dimensions for all other objects
   - Shows measurements overlaid on the image

### Fruit Classification Model Training

1. **Open the training notebook:**

   ```bash
   jupyter notebook Train.ipynb
   ```

2. **Follow the notebook to:**
   - Load and explore the fruit dataset
   - Set up data generators with augmentation
   - Build and compile the CNN model
   - Train the model with validation
   - Save the trained model as `Fruits_model.h5`

### Fruit Classification Testing

1. **Open the testing notebook:**

   ```bash
   jupyter notebook Test.ipynb
   ```

2. **Run the cells to:**
   - Load the pre-trained model (`Fruits_model.h5`)
   - Test on sample images
   - Get classification results with confidence scores

## 📊 Dataset Information

### Fruit Classification Dataset

- **Total Classes:** 5 (Apple, Banana, Mango, Orange, Pineapple)
- **Structure:** Organized in train/test split
- **Format:** JPEG/PNG images
- **Organization:** Each class has its own subdirectory
- **Usage:** Training data for CNN model development

## 📊 Model Information

### Fruit Classification Model

- **Architecture:** Convolutional Neural Network (CNN)
- **Input Size:** 150×150×3 (RGB images)
- **Classes:** 5 fruit categories (Apple, Banana, Mango, Orange, Pineapple)
- **Format:** TensorFlow/Keras (.h5)
- **Preprocessing:** Normalization (pixel values ÷ 255)
- **Training Features:**
  - Data augmentation with ImageDataGenerator
  - Dropout layers for regularization
  - Adam optimizer
  - Sparse categorical crossentropy loss

## 🔧 Technical Details

### Object Measurement Algorithm

1. **Image Preprocessing:**

   - Resize for performance optimization (1/5 scale)
   - Convert to grayscale
   - Apply Gaussian blur for noise reduction

2. **Edge Detection:**

   - Canny edge detection with thresholds (50, 100)
   - Morphological operations (dilation/erosion)

3. **Object Detection:**

   - Contour finding and filtering (minimum area: 100 pixels)
   - Minimum area rectangle calculation
   - Perspective correction and point ordering

4. **Measurement Calculation:**
   - Pixel-to-metric ratio calibration using reference object
   - Euclidean distance calculation between midpoints
   - Real-world dimension conversion to millimeters

### Fruit Classification Pipeline

1. **Data Loading:** Organized dataset structure with train/test splits
2. **Data Augmentation:** Rotation, zoom, horizontal flip during training
3. **Model Architecture:**
   - Convolutional layers with ReLU activation
   - MaxPooling for downsampling
   - Dropout for regularization
   - Dense layers for classification
4. **Training:** Supervised learning with labeled fruit images
5. **Inference:** Image preprocessing → Model prediction → Confidence scores

## 📝 Configuration Parameters

### Object Measurement

- `ref_obj_width`: Reference object width in millimeters
- `image_path`: Path to the input image
- Canny edge detection thresholds: (50, 100)
- Gaussian blur kernel: (7, 7)
- Minimum contour area: 100 pixels

### Fruit Classification

- `IMG_SIZE`: 150 pixels (model input requirement)
- Supported formats: JPG, PNG
- Color space: RGB

## 🎯 Use Cases

### Object Dimension Measurement

- Quality control in manufacturing and production
- Dimensional analysis of parts and components
- Educational demonstrations of computer vision techniques
- Automated measurement systems for inventory
- Research applications requiring precise measurements

### Fruit Classification

- Agricultural applications and crop sorting
- Food industry quality control and classification
- Educational machine learning and computer vision projects
- Retail automation and inventory management
- Research in agricultural technology and food processing

## 📈 Getting Started

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd Prof_Project
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **For Object Measurement:**

   ```bash
   python main.py
   ```

4. **For Fruit Classification:**
   ```bash
   jupyter notebook Train.ipynb  # To train a new model
   jupyter notebook Test.ipynb   # To test existing model
   ```

## 🚨 Limitations

### Object Measurement

- Requires objects to be on a flat surface
- Reference object must be clearly visible
- Works best with well-defined edges
- Lighting conditions affect accuracy

### Fruit Classification

- Limited to 5 predefined fruit types
- Requires good lighting and clear images
- Performance depends on image quality
- May struggle with partially visible fruits
