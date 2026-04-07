"""
CS535 - Leveraging Federated Learning for Multi-Label Thoracic Pathology Classification
"""
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import pandas as pd
import numpy as np
import random
import glob
import os
import io

# ______________________________________________________________________________
# hospitals_distribution

def get_hospital_dfs(data_path, train_filenames, eval_filenames):
    """
    Get train, eval, and combined DataFrames from specified hospital data path.
    """
    train_files = glob.glob(os.path.join(data_path, train_filenames))
    eval_files = glob.glob(os.path.join(data_path, eval_filenames))

    df_train = pd.concat([pd.read_parquet(f) for f in train_files], ignore_index=True)
    df_eval = pd.concat([pd.read_parquet(f) for f in eval_files], ignore_index=True)

    return df_train, df_eval, pd.concat([df_train, df_eval])

def plot_gender_distribution(gender_df):
    """
    Plot Gender Distribution of specified hospital `gender_df` DataFrame.
    """
    gender_df = gender_df.map({'M': 'Male', 'F': 'Female'})
    sns.countplot(x=gender_df, hue=gender_df, palette='colorblind', order=np.unique(gender_df))
    plt.title('Gender Distribution')
    plt.xlabel('Patient Gender')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def plot_age_distribution_by_gender(age_df, gender_df):
    """
    Plot Age and Gender Distribution given `age_df` and `gender_df` DataFrames.
    """
    gender_df = gender_df.map({'M': 'Male', 'F': 'Female'})

    combined_df = pd.DataFrame({
        'Age': age_df,
        'Gender': gender_df
    })
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(age_df, bins=20, kde=True, color=sns.color_palette('colorblind')[3], ax=ax[0])
    ax[0].set_title('Age Distribution')
    ax[0].set_xlabel('Patient Age')
    ax[0].set_ylabel('Frequency')
    
    sns.histplot(data=combined_df, x='Age', bins=20, kde=True, hue='Gender', palette='colorblind', multiple='stack', ax=ax[1])
    ax[1].set_title('Age Distribution by Gender')
    ax[1].set_xlabel('Patient Age')
    ax[1].set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def plot_view_distribution(view_df):
    """
    Plot View Position (AP/PA) distribution of hospital specified in `view_df` DataFrame.
    """
    sns.countplot(x=view_df, hue=view_df, palette='colorblind', order=np.unique(view_df))
    plt.title('View Distribution')
    plt.xlabel('View Position')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def plot_pathology_distribution(pathology_df):
    """
    Plot pathology distribution within hospital.
    """
    pathology_df = pathology_df.explode()
    frequencies = pathology_df.value_counts()
    
    sns.countplot(x=pathology_df, hue=pathology_df, palette='colorblind', order=frequencies.index)
    plt.title('Thoracic Pathology Distribution')
    plt.xlabel('Pathology Type')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_all_distributions(age_df, gender_df, view_df, pathology_df):
    """
    Plot all distributions (age, gender, view position, pathology) of hospital.    """
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax = ax.flatten()
    
    # Gender
    gender_df = gender_df.map({'M': 'Male', 'F': 'Female'})
    sns.countplot(x=gender_df, hue=gender_df, palette='colorblind', order=np.unique(gender_df), ax=ax[0])
    ax[0].set_title('Gender Distribution')
    ax[0].set_xlabel('Patient Gender')
    ax[0].set_ylabel('Frequency')

    # Age
    sns.histplot(age_df, bins=20, kde=True, color=sns.color_palette('colorblind')[3], ax=ax[1])
    ax[1].set_title('Age Distribution')
    ax[1].set_xlabel('Patient Age')
    ax[1].set_ylabel('Frequency')

    # Age by Gender
    combined_df = pd.DataFrame({
        'Age': age_df,
        'Gender': gender_df
    })
    sns.histplot(data=combined_df, x='Age', bins=20, kde=True, hue='Gender', palette='colorblind', multiple='stack', ax=ax[2])
    ax[2].set_title('Age Distribution by Gender')
    ax[2].set_xlabel('Patient Age')
    ax[2].set_ylabel('Frequency')

    # View Position
    sns.countplot(x=view_df, hue=view_df, palette='colorblind', order=np.unique(view_df), ax=ax[3])
    ax[3].set_title('View Distribution')
    ax[3].set_xlabel('View Position')
    ax[3].set_ylabel('Frequency')

    # Pathology
    pathology_df = pathology_df.explode()
    frequencies = pathology_df.value_counts()
    
    sns.countplot(x=pathology_df, hue=pathology_df, palette='colorblind', order=frequencies.index, ax=ax[4])
    ax[4].set_title('Thoracic Pathology Distribution')
    ax[4].set_xlabel('Pathology Type')
    ax[4].set_ylabel('Frequency')
    ax[4].tick_params(axis='x', rotation=90)

    fig.delaxes(ax[5])
    plt.tight_layout()
    plt.show()

# ______________________________________________________________________________
# x_ray_visualization.py

def get_train_df(data_path, train_filenames, name):
    """
    Retrieve a hospital's training data and store in DataFrame
    """
    train_files = glob.glob(os.path.join(data_path, train_filenames))
    df_train = pd.concat([pd.read_parquet(f) for f in train_files], ignore_index=True)
    df_train['hospital'] = name
    return df_train

def get_random_x_rays(hospitals, size):
    """
    Get `size` X-Rays from each hospital and return a dictionary where key: hospital name, value: (images_dict, labels)
    """
    x_rays_dict = {}
    for hospital in hospitals:
        random_indices = random.sample(hospital.index.tolist(), size)
        imgs_dict = hospital['image'].loc[random_indices].tolist()
        labels = hospital['label'].loc[random_indices].tolist()
        x_rays_dict[hospital['hospital'].unique()[0]] = (imgs_dict, labels)

    return x_rays_dict

def plot_x_rays(x_rays_dict, size):
    """
    Plot `size` X-Rays per Hospital. Each row corresponds to the X-Rays from one hospital.
    """
    nrows = len(x_rays_dict) # Just 3 hospitals
    ncols = size # Num. of X-Rays per hospital
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 4))
    for i, key in enumerate(x_rays_dict):
        images_dict, labels = x_rays_dict[key]
        for j, (img_bytes, label) in enumerate(zip(images_dict, labels)):
            img = Image.open(io.BytesIO(img_bytes['bytes']))
            ax[i, j].imshow(img, cmap='gray')
            ax[i, j].set_xlabel(label, fontsize=15)
            ax[i, j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax[i, ncols // 2].set_title(key, fontsize=20)
    plt.tight_layout()
    plt.show()