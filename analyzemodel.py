import io
import os
import random
import joblib
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA

if os.path.exists('models/svm_model.pkl'):
    label_encoder = joblib.load('models/label_encoder.pkl')

def analyze_model():
    global label_encoder
    data = np.load('models/features.npy')
    labels = np.load('models/labels.npy')

    # Perform PCA to reduce to 2 dimensions for visualization
    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(data)
    
    # Plot the data points and decision boundaries
    plt.figure(figsize=(22, 12))
    random_color = generate_random_color()
    name_index = 0
    plt.scatter(transformed_data[0, 0], transformed_data[0, 1], label=label_encoder.inverse_transform([name_index])[0], c=random_color)
    
    for i in range(len(labels)):
        if labels[i] != label_encoder.inverse_transform([name_index])[0]:
            random_color = generate_random_color()
            name_index += 1
            plt.scatter(transformed_data[i, 0], transformed_data[i, 1], label=label_encoder.inverse_transform([name_index])[0], c=random_color)
        else:
            plt.scatter(transformed_data[i, 0], transformed_data[i, 1], c=random_color)
    
    # Set the plot properties
    plt.xlim(transformed_data[:, 0].min() - 0.1, transformed_data[:, 0].max() + 0.3)
    plt.ylim(transformed_data[:, 1].min() - 0.1, transformed_data[:, 1].max() + 0.1)
    plt.grid(True)
    plt.legend()
    plt.xlabel('PCA Component Y')
    plt.ylabel('PCA Component X')
    plt.title('FaceLog SVM Model Scatter Plot')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.savefig('scatter_plot.png')

def generate_random_color():
    r = random.randint(0, 200)
    g = random.randint(0, 200)
    b = random.randint(0, 200)
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

analyze_model()