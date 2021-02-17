#import minisom
from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from pylab import plot, axis, show,pcolor,colorbar,bone

def som_with_images():
    for i in ["house.ppm","Pink_flower.jpg"]:
        img = plt.imread(i)
        plt.imshow(img)
        pixels = np.reshape(img, (img.shape[0]*img.shape[1], 3))/255
        som = MiniSom(x= 2, y = 2, input_len = 3, sigma=0.1, learning_rate=0.2)
        som.random_weights_init(pixels)

        starting_weights = som.get_weights().copy()
        som.train_random(pixels, 500)
        qnt = som.quantization(pixels)
        clustered = np.zeros(img.shape)

        for i, q in enumerate(qnt):
            clustered[np.unravel_index(i, dims=(img.shape[0], img.shape[1]))] = q

        plt.figure(figsize=(12, 6))
        plt.subplot(221)
        plt.title('Original')
        plt.imshow(img)
        plt.subplot(222)
        plt.title('Result')
        plt.imshow(clustered)
        plt.subplot(223)
        plt.title('Initial Colors')
        plt.imshow(starting_weights)
        plt.subplot(224)
        plt.title('Learnt Colors')
        plt.imshow(som.get_weights())
        plt.tight_layout()
        plt.show()

 
def som_with_data():
    data = np.genfromtxt(
        "breast-cancer-wisconsin.data.txt",
        delimiter=",",
        names=True,
        )

# Define what columns are considered features for
# describing the cells
    feature_names = [
        "clump_thickness",
        'uniform_cell_size',
        'uniform_cell_shape',
        'marginal_adhesion',
        'single_epi_cell_size',
        'bare_nuclei',
        'bland_chromation',
        'normal_nucleoli',
        'mitoses'
        ]

# Gather features into one matrix (this is the "X" matrix in task)
    features = []
    for feature_name in feature_names:
        features.append(data[feature_name])
    features = np.stack(features, axis=1)
        
        # Do same for class
    classes = data["class"]
        
        # Remove non-numeric values (appear as NaNs in the data).
        # If any feature in a sample is nan, drop that sample (row).
    cleaned_features = []
    cleaned_classes = []
    for i in range(features.shape[0]):
        if not np.any(np.isnan(features[i])):
            cleaned_features.append(features[i])
            cleaned_classes.append(classes[i])
    cleaned_features = np.stack(cleaned_features, axis=0)
    cleaned_classes = np.array(cleaned_classes)
                
                # Rename to match the exercises
    X = cleaned_features
    y = cleaned_classes
                
                # Standardize features with standard scaling ([x - mean] / std)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
                
                # Transform y into {0,1} label array.
                # Turn 2 into 0 and 4 into 1
    y = (y == 4).astype(np.int64)
                
                
    data,target = X,y
    som = MiniSom(x = 30, y=20, input_len = data.shape[1], sigma=1, learning_rate=0.5)
    som.random_weights_init(data)
                
    som.train_random(data, 500)
    bone()
    pcolor(som.distance_map().T)
    colorbar()
    markers = ['o','s','D']
    colors = ['r','g','b']
    for cnt,xx in enumerate(data):
        w = som.winner(xx)
        plot(w[0]+.5,w[1]+.5,markers[target[cnt]],markerfacecolor='None',
             markeredgecolor = colors[target[cnt]],markersize=12,markeredgewidth=2)
    axis([0,som._weights.shape[0],0,som._weights.shape[1]])
    show()
 
som_with_images()
som_with_data()