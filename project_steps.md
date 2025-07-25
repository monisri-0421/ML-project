# ML Project - Step-by-Step Process

## Step 1: Project Overview
This project is a binary image classification task where we use a Convolutional Neural Network (CNN) to distinguish between cat and dog images.

## Step 2: Data Collection
dataset: kaggle
- Downloaded the dataset (train.zip)
- Extracted images
- Divided them into training and validation sets

## Step 3: Data Preprocessing
- Resized all images to 128x128
- Converted images to NumPy arrays
- Normalized pixel values to range [0, 1]
- Augmented training set with flips, zoom, and rotations

## Step 4: Model Building
CNN model was built using tensorflow/keras with layers:
- conv2d,relu,maxpooling,dropout(to prevent over fitting)
- flatten and dense (for classification)
- final output layer with sigmoid activation
  model outputs 1 if image is dog and 0 if image is cat

## Step 5: Training the Model
Training configuration:
-Loss function: Binary Crossentropy
-Optimizer: Adam
-Epochs: 10–15
-Batch size: 32

## Step 6: Saving the Model
after training the model is saved as
-model.save('cat_dog_classifier.h5')

## Step 7: Making Predictions
To test the model:
-Loaded the saved .h5 model and used OpenCV to read a test image
-Preprocessed the image (resize + normalize)
-Predicted the class using model.predict()
Output was either cat or dog based on threshold > 0.5.

## Step 8: GitHub Repo Structure
ML-project/
│
├── notebooks/         # Jupyter Notebooks for training and testing
├── models/            # Saved trained models (.h5 files)
├── images/            # Screenshots for documentation
├── PROJECT_STEPS.md   # Step-by-step process (this file)
├── README.md          # Main project overview
├── requirements.txt   # Python dependencies


## Step 9: Deployment (Optional)
we built a fast api that:
-Accepts image uploads via an endpoint,Loads the saved model,Returns predicted class as JSON
-It was deployed using Render.com, a cloud service for hosting web apps.

## Step 10: Final Notes
The model performs well on clean images and can be improved with more training data and regularization
To run the project:
-Install dependencies from requirements.txt
-Run the training notebook
-Use the FastAPI app for predictions
