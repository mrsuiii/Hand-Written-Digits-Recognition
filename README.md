# Handwritten Digit Recognition using PyTorch

This project implements a **Convolutional Neural Network (CNN)** to recognize handwritten digits from the **MNIST dataset**. The notebook demonstrates:

- Loading and visualizing the MNIST dataset.
- Defining a CNN architecture with 3 convolutional layers and 2 fully connected layers.
- Training and testing the model using PyTorch.

## Dataset

The model uses the MNIST dataset, a collection of 70,000 grayscale images of handwritten digits (0-9) split into:

- 60,000 training images
- 10,000 testing images

Each image has a resolution of **28x28 pixels**.

## Model Architecture

The implemented CNN consists of:

1. **Convolutional Layers**:

   - 3 convolutional layers with ReLU activations.
   - Max pooling after each convolutional layer for downsampling.

2. **Fully Connected Layers**:

   - Two fully connected layers with ReLU and Softmax activation for classification.

### Summary

- **Input**: 28x28 grayscale image.
- **Output**: Probability distribution over 10 classes (digits 0-9).

## Visualization

The notebook includes code to visualize:

- Random samples from the MNIST dataset.
- Loss curves during training.
- Model predictions on test data.

## Dependencies

This project requires:

- `torch`
- `torchvision`
- `matplotlib`
- `numpy`
- `torchviz`

Install the required packages using:

```bash
pip install torch torchvision matplotlib numpy torchviz
```

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/handwritten-digit-recognition
   ```

2. Navigate to the project directory:

   ```bash
   cd handwritten-digit-recognition
   ```

3. Open the Jupyter Notebook:

   ```bash
   jupyter notebook HandWrittenDigitsRecognition.ipynb
   ```

4. Run all cells in the notebook to:

   - Load and visualize the MNIST dataset.
   - Train the CNN model.
   - Test the model's accuracy.

## Results

- The model achieves a high accuracy on the MNIST test set, demonstrating its ability to generalize to unseen data.
- Example predictions and their corresponding ground truth labels are displayed in the notebook.

## Project Structure

```
.
├── HandWrittenDigitsRecognition.ipynb  # Main notebook
├──  README.md                           # Project description
└── Architecture                          # Describe Architecture of the model 
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- **PyTorch**: For providing an easy-to-use deep learning framework.
- **MNIST Dataset**: A widely-used dataset for benchmarking machine learning models.

Feel free to fork, contribute, or provide feedback!

