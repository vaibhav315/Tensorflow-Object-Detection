**Face Mask Detection with TensorFlow and KerasCV

This project is an end-to-end implementation of an object detection pipeline for identifying face masks in images. Using the Kaggle "Face Mask Detection" dataset, the model is trained to classify faces into three categories: 'with_mask,' 'without_mask,' and 'mask_weared_incorrect.' This repository serves as a practical demonstration of applying deep learning techniques for a real-world computer vision task.

Key Features and Technical Details

Dataset Integration: The script automatically downloads and processes the Kaggle dataset, handling the extraction and organization of image and annotation files.

Data Parsing: It includes a robust parser for Pascal VOC XML annotations, converting the bounding box information and class labels into a clean pandas DataFrame. This structured approach simplifies the data handling process for model training.

TensorFlow Data Pipeline (tf.data): A high-performance data pipeline is constructed to efficiently load and preprocess images and their corresponding bounding boxes. The pipeline resizes and pads images to a uniform size and normalizes pixel values, preparing them for the neural network.

Model Architecture: The core of the project is a RetinaNet model, an anchor-based object detector known for its efficiency and accuracy. It leverages a ResNet50 backbone for feature extraction, which is a powerful and widely-used convolutional neural network.

Training and Evaluation: The model is trained using standard object detection losses (focal for classification and smoothl1 for regression). The training process includes callbacks for early stopping and model checkpointing to save the best-performing weights.

Inference & Visualization: The repository includes code to run inference on new images and visualize the detected bounding boxes with their corresponding class labels, providing a clear demonstration of the model's performance.**
