### Img display ###
import os, io
from matplotlib.cbook import to_filehandle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf



def img_pltsave(data, savepath="", tensorboard=False, all=False):
    
    if not isinstance(data, list):
       data = [data]
    for i in range(len(data)):
        if len(data[i].shape)>4:
             data[i] = data[i][0]
        if len(data[i].shape)<4:
            data[i] = tf.expand_dims(data[i], axis=0)
        elif (data[i].shape[0])>6:
            if not all:
                data[i] = data[i][::int((data[i].shape[0])/5)]


    rows_data = len(data)
    data_shape = data[0].shape
    columns_data = data_shape[0]

    aspect_ratio = data_shape[1] / data_shape[2]
    #aspect_ratio = data_shape[0] / data_shape[1]
    size_multiplier = 2
    nrow = rows_data*aspect_ratio*size_multiplier
    ncol = columns_data*size_multiplier
    figure, axarr = plt.subplots(
        rows_data,
        columns_data,
        gridspec_kw=dict(
            hspace=0.02,
            wspace=0.01, 
            top=1. - 0.5 / (nrow + 1), 
            bottom=0.5 / (nrow + 1),
            left=0.5 / (ncol + 1), 
            right=1. - 0.5 / (ncol + 1)),
        figsize=(ncol + 1, nrow + 1), 
    )
    for i in range(rows_data):
        for j in range(columns_data):
            if columns_data>1 and rows_data>1:
                #axarr[i, j].imshow(data[i][:, :, j], cmap="gray")
                axarr[i, j].imshow(data[i][j], cmap="gray")
                axarr[i, j].axis("off")
            elif rows_data == 1 & columns_data == 1:
                axarr.imshow(data[0][0], cmap="gray")    
                axarr.axis("off")
            elif columns_data>1:
                axarr[j].imshow(data[i][j], cmap="gray")
                axarr[j].axis("off")
            elif rows_data>1:
                axarr[i].imshow(data[i][j], cmap="gray")
                axarr[i].axis("off")
    if savepath:       
            print(f"saving image as {savepath}.png")
            plt.savefig(f"{savepath}.png", format='png')
            plt.close(figure)
    elif tensorboard:
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(figure)
            buf.seek(0)
            # Convert PNG buffer to TF image
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            # Add the batch dimension
            image = tf.expand_dims(image, 0)

            return image
    else:
        plt.show()
        plt.close()
    


def patch_pltsave(patches, ksizes, savepath="", patch_depth = 32,channels=3):

    patch_count = int(np.sqrt(patches.shape[0]))
    patch_height, patch_width = ksizes
    plt_shape = tf.reshape(patches[0], (patch_depth ,patch_height, patch_width, channels)).shape
    plt_count = int(plt_shape[0]/5)

    nrow = patch_count
    ncol = patch_count*plt_count
    size_multiplier = 5
    aspect_ratio = plt_shape[1] / plt_shape[2]
    n_plot_rows = 1
    row_shape = n_plot_rows*aspect_ratio*size_multiplier
    col_shape = plt_count*size_multiplier


    gs = gridspec.GridSpec(nrow, ncol,
            wspace=0.0, hspace=0.0, 
            top=1.-0.5/(row_shape+1), bottom=0.5/(row_shape+1), 
            left=0.5/(col_shape+1), right=1-0.5/(col_shape+1)) 
    fig = plt.figure(figsize=(col_shape+1, row_shape+1))  
            
    k = 0
    for i in range(nrow):
        for j in range(nrow):
                patch_img = tf.reshape(patches[k], (patch_depth,patch_height, patch_width, channels))
                patch_img = patch_img[::int((patch_img.shape[0])/5)]
                k = k+1

                for nc in range(plt_count):
                    ax= plt.subplot(gs[i,j+(patch_count*nc)])
                    ax.imshow(patch_img[nc], cmap='gray')
                    ax.axis('off')

    if savepath:
        print(f"saving image to {savepath}")
        plt.savefig(f"{savepath}.png")
        plt.clf()
    else:
        plt.show()
        plt.close()



