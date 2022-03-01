
### Img display ###
import matplotlib.pyplot as plt

def display(data):
    """
    Given a list of images, display them in a grid.
    
    :param data: the data to be displayed
    """

    if not isinstance(data, list):
       data = [data]
       img_list = False
    
    columns_data = len(data)
    rows_data = max([data[i].shape[0] for i in range(columns_data)])

    # width_px = max([data[i].shape[1] for i in range(columns_data)])
    # height_px = max([data[i].shape[2] for i in range(columns_data)])

    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(10,60), #tune this 
    )
    for i in range(columns_data):
        for j in range(rows_data):
            if not isinstance(img_list, list):
                axarr[j].imshow(data[j, :, :], cmap="gray")    
                axarr[j].axis("off")
            else:
                axarr[j, i].imshow(data[i][j, :, :], cmap="gray")
                axarr[j, i].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()
    
##display([train_data[0,0],train_data[1,0]])
#display([x_train[0][0],x_train[0][1]])
