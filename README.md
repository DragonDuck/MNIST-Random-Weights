

```python
import keras
import keras.models as kmod
import keras.layers as klay
import keras.utils as kutils
import numpy as np
import sklearn.ensemble
import sklearn.preprocessing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

    Using TensorFlow backend.


# Classifying MNIST with Random-Weight Features
The MNIST dataset is a common "benchmark" dataset for deep learning tutorials and research, e.g. exploring new network architectures and network elements. However, the dataset itself is unsuitable for this task as it is too trivial. In this notebook, I show that classifying the digits of this dataset with a deep convolutional neural network (DCNN) is essentially a trivial task. For this, I compare the classification accuracy of a conventionally trained DCNN to the features extracted by an untrained, i.e. randomly initialized, DCNN. This comparison will highlight that training a neural network is unnecessary to accurately differentiate MNIST digits.

Deep neural networks can be interpreted as consisting of two parts: feature extraction and classification/regression. The assignment of layers to roles is open to interpretation, but a simple interpretation is that the output layer is the classification/regression layer and all previous hidden layers are responsible for feature extraction.

![Hidden Layers](dnn_hidden_layers.png)
<center>Image taken from <a href="https://towardsdatascience.com/applied-deep-learning-part-1-artificial-neural-networks-d7834f67a4f6">TowardsDataScience</a></center>

DCNNs, in particular, can be interpreted as extracting hierarchical features with convolutional and pooling layers that are then classified/regressed on in the final, fully connected output layers.

![Hierarchical Features](dcnn_feature_extraction.png)
<center>Image adapted from <a href="https://www.analyticsvidhya.com/blog/2017/04/comparison-between-deep-learning-machine-learning/">AnalyticsVidhya</a></center>

## Load Data


```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Min-Max Scaling
train_min = np.min(x_train, axis=(1, 2))[:, np.newaxis, np.newaxis]
train_max = np.max(x_train, axis=(1, 2))[:, np.newaxis, np.newaxis]
x_train = (x_train - train_min) / (train_max - train_min)
test_min = np.min(x_test, axis=(1, 2))[:, np.newaxis, np.newaxis]
test_max = np.max(x_test, axis=(1, 2))[:, np.newaxis, np.newaxis]
x_test = (x_test - test_min) / (test_max - test_min)

# Transform labels
y_train_cat = kutils.to_categorical(y_train)
y_test_cat = kutils.to_categorical(y_test)

# Transform input to be 4D
x_train = x_train[..., None]
x_test = x_test[..., None]
```

## Conventional DCNN with Training
For the first step, a generic DCNN is constructed and all layers are included in the training. The network architecture chosen here has no deeper meaning and was chosen for its simplicity. Furthermore, this is not meant to be a study in best practices as I only wish to highlight the importance of training the feature extraction layers.

The network is validated on the test subset of the MNIST dataset, which is not included during the training.


```python
# create model
model = kmod.Sequential()

# add model layers
model.add(klay.Conv2D(
    filters=32, kernel_size=3, activation='relu', 
    input_shape=x_train.shape[1:]))
model.add(klay.MaxPool2D())
model.add(klay.Conv2D(
    filters=32, kernel_size=3, activation='relu', 
    kernel_initializer="glorot_uniform"))
model.add(klay.MaxPool2D())
model.add(klay.Conv2D(
    filters=32, kernel_size=3, activation='relu', 
    kernel_initializer="glorot_uniform"))
model.add(klay.Flatten())
model.add(klay.Dense(
    units=10, activation='softmax', 
    kernel_initializer="glorot_uniform"))

# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train_cat, validation_data=(x_test, y_test_cat), epochs=5)
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/5
    60000/60000 [==============================] - 21s 354us/step - loss: 0.1868 - acc: 0.9423 - val_loss: 0.0626 - val_acc: 0.9792
    Epoch 2/5
    60000/60000 [==============================] - 20s 335us/step - loss: 0.0609 - acc: 0.9809 - val_loss: 0.0371 - val_acc: 0.9874
    Epoch 3/5
    60000/60000 [==============================] - 20s 338us/step - loss: 0.0437 - acc: 0.9862 - val_loss: 0.0370 - val_acc: 0.9883
    Epoch 4/5
    60000/60000 [==============================] - 22s 366us/step - loss: 0.0354 - acc: 0.9885 - val_loss: 0.0302 - val_acc: 0.9899
    Epoch 5/5
    60000/60000 [==============================] - 25s 424us/step - loss: 0.0283 - acc: 0.9912 - val_loss: 0.0306 - val_acc: 0.9906





    <keras.callbacks.History at 0x12ef4cd30>



The network is, as expected, extremely accurate in classifying handwritten digits.


```python
cnn_acc = model.evaluate(x=x_test, y=y_test_cat, verbose=False)
print("Trained CNN Accuracy = {}%".format(np.round(cnn_acc[1]*100, 2)))
```

    Trained CNN Accuracy = 99.06%


## Extracting Features with Random Network Weights
In this second step, only the final classification layer is trained. All hidden layers are randomly initialized (Glorot uniform initialization with default parameters) and kept that way. This network will be trained and evaluated several times to identify the stability of these random-weight features.

Note that this is essentially an [Extreme Learning Machine](https://en.wikipedia.org/wiki/Extreme_learning_machine).


```python
random_feature_accuracies = []
random_feature_models = []

for ii in range(10):
    # create model
    model = kmod.Sequential()

    # add model layers
    model.add(klay.Conv2D(
        filters=32, kernel_size=3, activation='relu', 
        input_shape=x_train.shape[1:], 
        trainable=False))
    model.add(klay.MaxPool2D(trainable=False))
    model.add(klay.Conv2D(
        filters=32, kernel_size=3, activation='relu', 
        kernel_initializer="glorot_uniform", 
        trainable=False))
    model.add(klay.MaxPool2D(trainable=False))
    model.add(klay.Conv2D(
        filters=32, kernel_size=3, activation='relu', 
        kernel_initializer="glorot_uniform", 
        trainable=False))
    model.add(klay.Flatten(trainable=False))
    model.add(klay.Dense(
        units=10, activation='softmax', 
        kernel_initializer="glorot_uniform"))

    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train_cat, validation_data=(x_test, y_test_cat), epochs=5, verbose=0)
    random_feature_accuracies.append(model.evaluate(x=x_test, y=y_test_cat, verbose=0)[1])
    random_feature_models.append(model)
```

The accuracy of the partially trained DCNN is slightly worse than that of the fully trained network but very stable across iterations.


```python
print("Random Feature Accuracy = ({} +/- {})%".format(
    np.round(np.mean(random_feature_accuracies)*100, 2), 
    np.round(np.std(random_feature_accuracies)*100, 2)))
```

    Random Feature Accuracy = (91.05 +/- 0.94)%



```python
random_feature_accuracies_df = pd.DataFrame({
    "Iteration": range(10), 
    "Accuracy": random_feature_accuracies})
myFigure = sns.catplot(
    x="Iteration", y="Accuracy", data=random_feature_accuracies_df, 
    kind="bar", color="skyblue");
myFigure.ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda num, tick_num: "{:.0f}%".format(num*100)))
```


![png](output_12_0.png)


### Alternative Classifiers
The single-layer classification step is relatively simple. We can also look at what happens if we classify the random-weight features, extracted from the second-to-last layer of the DCNN above, with a more sophisticated method such as a random forest


```python
random_features = []
random_feature_forest_accs = []

for model in random_feature_models:
    # Extract features from submodel
    submodel = kmod.Sequential(layers=model.layers[:-1])
    features_train = submodel.predict(x_train)
    features_test = submodel.predict(x_test)
    random_features.append([features_train, features_test])
    
    # Random forest model
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
    rf.fit(X=features_train, y=y_train)
    random_feature_forest_accs.append(rf.score(X=features_test, y=y_test))
```

A random forest evaluated on the test data exhibits a high accuracy across all iterations.


```python
print("Random Forest Accuracy on Random-Weight Features = ({} +/- {})%".format(
    np.round(np.mean(random_feature_forest_accs)*100, 2), 
    np.round(np.std(random_feature_forest_accs)*100, 2)))
```

    Random Forest Accuracy on Random-Weight Features = (95.09 +/- 0.4)%



```python
random_feature_forest_accs_df = pd.DataFrame({
    "Iteration": range(10), 
    "Accuracy": random_feature_forest_accs})
myFigure = sns.catplot(
    x="Iteration", y="Accuracy", data=random_feature_forest_accs_df, 
    kind="bar", color="skyblue");
myFigure.ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda num, tick_num: "{:.0f}%".format(num*100)))
```


![png](output_17_0.png)


#### Unsupervised Clustering
Unsupervised clustering of the mean features shows a very clear separation of the individual classes. For this clustering, I used [t-Distributed Stochastic Neighbor Embedding (t-SNE)](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding). This method is well-suited to visualize clustering behaviour in high dimensions but this comes at the cost of a strong dependence on the parameters. This section serves only to visualize that random-weight features *can* be clustered, not to analyze the clustering behaviour itself.


```python
# Average test features over all iterations
all_test_features = np.mean(np.stack([s[1] for s in random_features]), axis=0)
all_train_features = np.mean(np.stack([s[0] for s in random_features]), axis=0)
```


```python
# Compute t-SNE
import sklearn.manifold
tsne_features = sklearn.manifold.TSNE().fit_transform(X=all_test_features)
tsne_features = pd.DataFrame(tsne_features, columns=("X", "Y"))
tsne_features["Label"] = y_test
tsne_features["Label"] = tsne_features["Label"].astype("str")
```


```python
sns.relplot(
    data=tsne_features, x="X", y="Y", hue="Label",
    palette=sns.color_palette(palette="bright", n_colors=10))
```




    <seaborn.axisgrid.FacetGrid at 0x1a34b1e7f0>




![png](output_21_1.png)


## Conclusion
The accuracy of a classifier trained on randomly initialized weights leads to comparable accuracies as a fully trained network when attempting. A simple non-linear classifier, as used in the semi-trained DCNN, shows accuracies notably better than random guessing (~10%) while a non-parametric, random forest classifier trained on the random-weight features will exhibit nearly equivalent accuracy as the fully trained DCNN. The random-weight features form distinct clusters that correspond to the digits they represent.

These results indicate that the MNIST dataset is too trivial to serve as a benchmark dataset. Any conclusions drawn from this dataset most likely reflect this triviality more than the power of the research matter, e.g. network architectures or novel network elements.
