import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def plot_image(imagematrix,nrow,ncol):
    '''show a signle image'''
    f, a = plt.subplots(1, 1, figsize=(1, 1))
    a.imshow(np.reshape(imagematrix,(nrow,ncol)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()

def plot_sequence_image(imageArr):
    fig = plt.figure()
    img_id = 0
    cur_img = imageArr[0]
    im = plt.imshow(cur_img, animated=True)
    print(img_id)
    def unpdate_img(*args):
        img_id = args[0]
        data_shape = np.shape(imageArr)
        img_num = data_shape[0]
        img_id += 1
        if img_id < img_num:
            cur_img = imageArr[img_id]
            im.set_array(cur_img)
            print(img_id)
        return im,
    ani = animation.FuncAnimation(fig, unpdate_img, fargs= [img_id],interval=100, blit=True, repeat=False)
    plt.show()


def display_image_sequence(x):
    img_num = len(x)
    amp_imgs = []
    for img_id in range(img_num):
        Cur_img = np.abs(x[img_id])
        Cur_img = np.reshape(Cur_img, [128, 128])
        print('Img_id = %d' % img_id)
        print(['Imgshape=', Cur_img.shape])
        amp_imgs.append(Cur_img)
    plot_sequence_image(amp_imgs)


def plot_curve(y_, x_coord = None,  fig_id =1):
    fig1 = plt.figure(fig_id)
    ax1 = fig1.add_subplot(111)
    if x_coord is None:
        data_num = len(y_)
        x_coord = np.arange(1, data_num+1 , 1, dtype=np.int32)
    ax1.plot(x_coord, y_, color='red')
    ax1.set_ylabel('cost values for training set')
    ax1.set_xlabel('epoch')

    ax1.set_title("Cost curve while training")
    fig1.show()
    plt.waitforbuttonpress()