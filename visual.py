import matplotlib.pyplot as plt
import pandas as pd


def plot_loss_curves():
    df = pd.read_csv('./results.csv')
    plt.figure()
    plt.plot(df['epoch'], df['accuracy'], label='training accuracy')
    plt.plot(df['epoch'], df['val_accuracy'], label='validation accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_loss_curves()