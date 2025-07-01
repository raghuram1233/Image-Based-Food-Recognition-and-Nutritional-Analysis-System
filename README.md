# Computer Vision Project

This repository contains two main computer vision applications: **Object Dimension Measurement** and **Fruit Classification**.

## 📁 Project Structure

```
Prof_Project/
├── main.py                 # Object dimension measurement script
├── fruits.ipynb           # Fruit classification notebook
├── Fruits_models.h5       # Pre-trained CNN model for fruit classification
├── imgs/                  # Sample images
│   ├── apple.jpg         # Sample fruit image
│   └── test1.png         # Sample object measurement image
└── README.md             # Project documentation
```

## 🎯 Features

### 1. Object Dimension Measurement (`main.py`)

- Measures real-world dimensions of objects in images
- Uses computer vision techniques for edge detection and contour analysis
- Provides measurements in millimeters using a reference object for scale calibration
- Automatic object detection and bounding box calculation

### 2. Fruit Classification (`fruits.ipynb`)

- Classifies fruit images using a pre-trained CNN model
- Supports 5 fruit categories: Apple, Banana, Grape, Mango, Strawberry
- Provides confidence scores for predictions
- Interactive Jupyter notebook interface

## 🛠️ Dependencies

### Required Libraries

```bash
# Core computer vision and machine learning
opencv-python>=4.5.0
tensorflow>=2.8.0
numpy>=1.21.0

# Additional utilities
scipy>=1.7.0
imutils>=0.5.4
```

### Installation

```bash
pip install opencv-python tensorflow numpy scipy imutils
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

### Fruit Classification

1. **Open the Jupyter notebook:**

   ```bash
   jupyter notebook fruits.ipynb
   ```

2. **Set the image path:**

   ```python
   File_Path = "imgs/apple.jpg"  # Path to your fruit image
   ```

3. **Run the cells to:**
   - Load the pre-trained model
   - Preprocess the image
   - Get classification results with confidence scores

## 📊 Model Information

### Fruit Classification Model

- **Architecture:** Convolutional Neural Network (CNN)
- **Input Size:** 150×150×3 (RGB images)
- **Classes:** 5 fruit categories
- **Format:** TensorFlow/Keras (.h5)
- **Preprocessing:** Normalization (pixel values ÷ 255)

## 🔧 Technical Details

### Object Measurement Algorithm

1. **Image Preprocessing:**

   - Resize for performance optimization
   - Convert to grayscale
   - Apply Gaussian blur for noise reduction

2. **Edge Detection:**

   - Canny edge detection
   - Morphological operations (dilation/erosion)

3. **Object Detection:**

   - Contour finding and filtering
   - Minimum area rectangle calculation
   - Perspective correction

4. **Measurement Calculation:**
   - Pixel-to-metric ratio calibration
   - Euclidean distance calculation
   - Real-world dimension conversion

### Fruit Classification Pipeline

1. **Image Loading:** OpenCV imread
2. **Preprocessing:** Resize to 150×150, normalize pixels
3. **Prediction:** CNN forward pass
4. **Output:** Probability distribution across fruit classes

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

- Quality control in manufacturing
- Dimensional analysis of parts and components
- Educational demonstrations of computer vision
- Automated measurement systems

### Fruit Classification

- Agricultural applications
- Food sorting and classification
- Educational machine learning projects
- Retail automation

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
