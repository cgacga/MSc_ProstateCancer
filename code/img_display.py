
### Img display ###
import matplotlib.pyplot as plt

def img_pltsave(data, savepath=""):
    """
    Given a list of images, display them in a grid.
    
    :param data: the data to be displayed
    """

    if not isinstance(data, list):
       data = [data]
        
    rows_data = len(data)
    if len(data[0].shape) >3:
        if (data[0].shape[-2])>6:
            for i in range(len(data)):
                data[i] = data[i][:,:,::int((data[i].shape[-2])/4)]
        columns_data = data[0].shape[-2]#max([data[i].shape[-2] for i in range(rows_data)])
    else: 
        columns_data = rows_data
        rows_data = data[0].shape[-1]#max([data[i].shape[-1] for i in range(rows_data)])

    from tensorflow.python.ops.numpy_ops import np_config
    np_config.enable_numpy_behavior()


    aspect_ratio = data[0].shape[0] / data[0].shape[1]
    nrow = rows_data*5
    ncol = columns_data*5*aspect_ratio
    f, axarr = plt.subplots(
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
    if rows_data >1 or columns_data >1 :
        for i in range(rows_data):
            for j in range(columns_data):
                if len(data[i].shape) >3 and len(data)>1:
                    axarr[i, j].imshow(data[i][:, :, j].transpose((1,0,2)), cmap="gray")
                    axarr[i, j].axis("off")
                elif len(data[i].shape) >3:
                    axarr[j].imshow(data[i][:, :, j].transpose((1,0,2)), cmap="gray")    
                    axarr[j].axis("off")
                else:
                    axarr[j].imshow(data[j][:, :, i].transpose((1,0,2)), cmap="gray")    
                    axarr[j].axis("off")
    else:
        axarr.imshow(data[0].transpose((1,0,2)), cmap="gray")    
        axarr.axis("off")

    if savepath:
        print(f"saving image to {savepath}")
        plt.savefig(savepath)
        plt.clf()
    else:
        plt.show()
    
# Single image
# img_pltsave([x_train_noisy[0][0,:,:,0]])
# Two images
# img_pltsave([x_train_noisy[0][0,:,:,0],x_train[0][0,:,:,0]])
# One image series
# img_pltsave([x_train_noisy[0][0]])
# Two image series
# img_pltsave([x_train_noisy[0][0],x_train[0][0]])
# Three image series
# img_pltsave([x_train_noisy[0][0],x_train[0][0],x_train[0][0]])


# import importlib
# import img_display
# importlib.reload(img_display)
# from img_display import *