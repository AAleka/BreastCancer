import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

np.random.seed(1337)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.set_option("display.max_column", 100)
pd.set_option("display.width", 1000)


def dataVisualization(data, start=0):
    if start == 0:
        plt.figure(figsize = (5,5))
        plt.hist(data['diagnosis'])
        plt.title('Diagnosis (M=malignant , B=benign)')

        plt.show()
    else:
        figures(data, start, start+10)


def figures(data, s, e):
    plt.rcParams.update({'font.size': 8})
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 10))
    axes = axes.ravel()

    features = list(data.columns[s:e])

    dataM = data[data['diagnosis'] == 'M']
    dataB = data[data['diagnosis'] == 'B']

    for idx, ax in enumerate(axes):
        ax.figure
        binwidth = (max(data[features[idx]]) - min(data[features[idx]])) / 50
        ax.hist([dataM[features[idx]], dataB[features[idx]]],
                bins=np.arange(min(data[features[idx]]), max(data[features[idx]]) + binwidth, binwidth),
                alpha=0.5, stacked=True, density=True, label=['M', 'B'], color=['r', 'g'])
        ax.legend(loc='upper right')
        ax.set_title(features[idx])

    plt.tight_layout()

    plt.show()


def dataPreparation(df):
    data = df.iloc[:, 1:32]

    data['diagnosis'] = data['diagnosis'].map(dict(M=int(1), B=int(0)))
    data = data.rename(columns={"diagnosis": "cancer"})
    data = data.drop(columns=['perimeter_mean', 'area_mean', 'perimeter_se', 'area_se', 'perimeter_worst', 'area_worst'])
    
    scaler = StandardScaler()

    data.iloc[:, 1:] = scaler.fit_transform(data.iloc[:, 1:])

    return data


def XYSplit(data):
    X = data.iloc[:, 1:]
    y = data['cancer']

    return X, y


def modelCreate():
    callback = EarlyStopping(monitor='val_loss', patience=17, restore_best_weights=True)
    model = Sequential()

    model.add(Dense(50, activation='relu', input_dim=24, kernel_regularizer='l2'))
    model.add(Dense(20, activation='relu', kernel_regularizer='l2'))
    model.add(Dense(1, activation='sigmoid', name='Output'))

    model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])

    return model, callback


def trainingPlot(history):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    ax1.plot(history.history['accuracy'], label='accuracy')
    ax1.plot(history.history['val_accuracy'], label='val_accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim([0.5, 1])
    ax1.legend(loc='lower right')

    ax2.plot(history.history['loss'], label='loss')
    ax2.plot(history.history['val_loss'], label='val_loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='lower right')

    plt.show()


def confusionMatrix(cmLR, scoreLR, cmSVM, scoreSVM, cmNN, scoreNN):
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.2)

    sns.heatmap(cmLR, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r', ax=ax1)
    ax1.set_ylabel('Actual label')
    ax1.set_xlabel('Predicted label')
    all_sample_title = 'Logistic Regression Score: {0}'.format(round(scoreLR, 5))
    ax1.title.set_text(all_sample_title)
    
    sns.heatmap(cmSVM, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r', ax=ax2)
    ax2.set_ylabel('Actual label')
    ax2.set_xlabel('Predicted label')
    all_sample_title = 'Support Vectore Machine Score: {0}'.format(round(scoreSVM, 5))
    ax2.title.set_text(all_sample_title)
    
    sns.heatmap(cmNN, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r', ax=ax3)
    ax3.set_ylabel('Actual label')
    ax3.set_xlabel('Predicted label')
    all_sample_title = 'Neural Network Score: {0}'.format(round(scoreNN, 5))
    ax3.title.set_text(all_sample_title)

    plt.show()
