import pandas as pd
from models import models

# Initialize the model with parameters
model = models(learning_rate=0.001, mse_threshold=0, epochs=85, bias=False)

# Read in the CSV file
model.read_csv("birds.csv")

# Verify that both categories are present in the data
unique_categories = model.data_frame['bird category'].unique()
if 'A' in unique_categories and 'B' in unique_categories:
    print("Both categories 'A' and 'B' found, proceeding with tests.")
    
    # Test the Perceptron model
    print("Testing Perceptron model:")
    perceptron_predictions = model.preceptron_model(0, 1, 'A', 'B')
    print("Perceptron Predictions:", perceptron_predictions)
  
