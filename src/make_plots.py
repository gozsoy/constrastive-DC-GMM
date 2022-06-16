import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
import seaborn as sns
import pandas as pd
import os
import numpy as np



def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--plot_name", type=str, help="Name used for charts", required=True)
    parser.add_argument("--log_dir", type=str, help="FQ path to embedding logs", required=True)

    args = parser.parse_args()
    return args


def load_data_from_fq_path(fq_path):
    fq_path = os.path.expanduser(fq_path)
    return tf.train.load_variable(fq_path, "embedding")


def load_labels_from_fq_path(fq_path):
    fq_path = os.path.expanduser(fq_path)
    return pd.read_table("{}/{}".format(fq_path, "metadata.tsv"), sep="\t", names=["labels"])


def load_results_data(fq_path):
    fq_path = os.path.expanduser(fq_path)
    return np.load("{}/conf_mat.npy".format(fq_path))


def main():
    # Parse arguments
    args = parse_args()

    # PCA
    data = load_data_from_fq_path(args.log_dir)
    standardized_data = StandardScaler().fit_transform(data)
    pca_obj = PCA(2)
    pca_obj.fit(standardized_data)
    pca_data = pca_obj.transform(standardized_data)
    labels = load_labels_from_fq_path(args.log_dir)["labels"]
    # automatize this as well
    label_map = ["0","1","2","3","4","5","6","7","8","9"]
    labels = labels.apply(lambda x: label_map[x])

    df = pd.DataFrame(
        {"PC1": pca_data[:, 0], "PC2": pca_data[:, 1], "Label": labels})

    figure = plt.figure(figsize=(8, 10))
    figure.set_rasterized(True)
    colours = sns.color_palette("tab10", len(labels.unique()))
    
    ax = sns.scatterplot(x="PC1", y="PC2", hue="Label", data=df, legend="full", alpha=0.3, palette=colours)
    plt.title(args.plot_name, fontsize=18)
    plt.xticks([], [])
    plt.yticks([], [])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_rasterized(True)
    plt.savefig(os.path.expanduser(args.log_dir) + "/PCA.png") # .eps
    plt.close()

    # Confusion matrix
    plt.figure(figsize=(8, 8))
    conf_mat = load_results_data(args.log_dir)
    ax = sns.heatmap(conf_mat, cmap="YlGnBu", annot=True, fmt="d")
    ax.set(xlabel="Predicted Cluster", ylabel="True Cluster")
    plt.title(args.plot_name + " confusion matrix")
    plt.savefig(os.path.expanduser(args.log_dir) + "/confusion_matrix.png") # .eps
    plt.close()


if __name__ == '__main__':
    main()