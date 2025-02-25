from matplotlib import pyplot as plt
import numpy as np
import wandb
from taufactor import metrics
from itertools import combinations

progress_dir = 'progress/'


def show_grey_image(image, title):
    """
    Plots the image in grey scale, assuming the image is 1 channel of 0-255
    """
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    wandb.log({title: [wandb.Image(plt)]})
    # plt.show()


def log_metrics(g_output, hr_metrics):
    """
    Logs the volume fraction and surface area metrics of the
    generated super-res volumes in wandb.
    :param g_output: the current output from g (batch_size*64^3 tensors)
    :param hr_metrics: the same metrics of the high-res 2D slice for
    comparison.
    """
    g_output = one_hot_decoding(fractions_to_ohe(g_output))
    # The super-res volume fraction and surface area values:
    sr_vf, sr_sa = vf_sa_metrics(g_output)
    hr_vf, hr_sa = hr_metrics
    if len(sr_vf) != len(hr_vf): # safety loop for the cases when for some reason any phase was not generated in Super-Resolution
        m_loss = 777
        return m_loss
    else:
        vf_labels, sa_labels = ["VF pore ", "VF QUARTZ ", "VF FS ", "VF CLAY "], \
                           ["SA pore/QUARTZ ", "SA pore/FS ", "SA pore/CLAY ", "SA QUARTZ/FS ", "SA QUARTZ/CLAY ", "SA FS/CLAY ",]
        [wandb.log({vf_labels[i] + 'SR': sr_vf[i]}) for i in range(len(sr_vf))]
        [wandb.log({vf_labels[i] + 'HR': hr_vf[i]}) for i in range(len(hr_vf))]
        m_loss = [np.abs(1-sr_vf[i]/hr_vf[i]) for i in range(len(hr_vf))]
        [wandb.log({sa_labels[i] + 'SR': sr_sa[i]}) for i in range(len(sr_sa))]
        [wandb.log({sa_labels[i] + 'HR': hr_sa[i]}) for i in range(len(hr_sa))]
        m_loss += [np.abs(1 - sr_sa[i] / hr_sa[i]) for i in range(len(hr_sa))]
        m_loss = np.mean(m_loss)
        wandb.log({'Metrics percentage difference': m_loss})
        return m_loss


def vf_sa_metrics(batch_images):
    """
    :param batch_images: a 4-dim or 3-dim array of images (batch_size x H x W or batch_size x D x H x W)
    :return: a list of the mean volume fractions of the different phases and
    the interfacial surface area between every pair of phases.
    """
    batch_size = batch_images.shape[0]
    phases = np.unique(batch_images)
    vf = np.mean([[(batch_images[j] == p).mean() for p in phases] for j in range(batch_size)], axis=0)
    sa = np.mean([[metrics.surface_area(batch_images[j], [ph1, ph2]).item() for ph1, ph2 in combinations(phases, 2)] for j in range(batch_size)], axis=0)
    return list(vf), list(sa)


def plot_fake_difference(images, save_dir, filename, with_deg=False):
    # first move everything to numpy
    # rand_sim = np.array(input_to_g[:, 2, :, :])
    images = [np.array(image) for image in images]
    images[1] = fractions_to_ohe(images[1])
    images[2] = fractions_to_ohe(images[2])  # the output from g needs to ohe
    if with_deg:
        images[3] = fractions_to_ohe(images[3])  # also the slices
    images = [one_hot_decoding(image) for image in images]
    save_three_by_two_grey(images, save_dir + ' ' + filename, save_dir,
                           filename, with_deg)


def save_three_by_two_grey(images, title, save_dir, filename, with_deg=False):
    if with_deg:
        f, axarr = plt.subplots(5, 3)
    else:
        f, axarr = plt.subplots(4, 2)
    for i in range(3):
        for j in range(2):
            length_im = images[i].shape[1]
            middle = int(length_im/2)
            axarr[i, j].imshow(images[i][j, middle, :, :], cmap='gray', vmin=0, vmax=3)
            axarr[i, j].set_xticks([0, length_im-1])
            axarr[i, j].set_yticks([0, length_im-1])
    for j in range(2):  # showing xy slices from 'above'
        axarr[3, j].imshow(images[2][j, :, :, 4], cmap='gray', vmin=0, vmax=3)
    if with_deg:
        for j in range(3):  # showing 45 deg slices
            axarr[4, j].imshow(images[3][j, :, :], cmap='gray', vmin=0, vmax=3)
    plt.suptitle(title)
    wandb.log({"running slices": plt})
    plt.savefig(progress_dir + save_dir + '/' + filename + '.png')
    plt.close()


def cbd_to_pore(im_with_cbd):
    """
    :return: the image without cbd. cbd -> pore.
    """
    res = np.copy(im_with_cbd)
    res[res == 255] = 0
    return res


def one_hot_encoding(image, phases):
    """
    :param image: a [depth, height, width] 3d image
    :param phases: the unique phases in the image
    :return: a one-hot encoding of image.
    """
    im_shape = image.shape
    res = np.zeros((len(phases), ) + im_shape, dtype=image.dtype)
    # create one channel per phase for one hot encoding
    for count, phase in enumerate(phases):
        image_copy = np.zeros(im_shape, dtype=image.dtype)  # just an encoding
        # for one channel
        image_copy[image == phase] = 1
        res[count, ...] = image_copy
    return res


def one_hot_decoding(image):
    """
    decodes the image back from one hot encoding to grayscale for
    visualization.
    :param image: a [batch_size, phases, height, width] tensor/numpy array
    :return: a [batch_size, height, width] numpy array
    """
    np_image = np.array(image)
    im_shape = np_image.shape
    phases = im_shape[1]
    res = np.zeros([im_shape[0]] + list(im_shape[2:]))

    # the assumption is that each pixel has exactly one 1 in its phases
    # and 0 in all other phases:
    for i in range(phases):
        if i == 0:
            continue  # the res is already 0 in all places..
        phase_image = np_image[:, i, ...]
        res[phase_image == 1] = i
    return res


def fractions_to_ohe(image):
    """
    :param image: a [n,c,w,h] image (generated) with fractions in the phases.
    :return: a one-hot-encoding of the image with the maximum rule, i.e. the
    phase which has the highest number will be 1 and all else 0.
    """
    np_image = np.array(image)
    res = np.zeros(np_image.shape)
    # Add a little noise for (0.5, 0.5) situations.
    np_image += (np.random.rand(*np_image.shape) - 0.5) / 100
    # finding the indices of the maximum phases:
    arg_phase_max = np.expand_dims(np.argmax(np_image, axis=1), axis=1)
    # make them 1:
    np.put_along_axis(res, arg_phase_max, 1, axis=1)
    return res


def graph_plot(data, labels, pth, filename):
    """
    simple plotter for all the different graphs
    :param data: a list of data arrays
    :param labels: a list of plot labels
    :param pth: where to save plots
    :param filename: the directory name to save the plot.
    :return:
    """

    for datum,lbl in zip(data,labels):
        plt.plot(datum, label = lbl)
    plt.legend()
    plt.savefig(progress_dir + pth + '/' + filename)
    plt.close()


def calc_and_save_eta(steps, time, start, i, epoch, num_epochs, filename):
    """
    Estimates the time remaining based on the elapsed time and epochs
    :param steps: number of steps in an epoch
    :param time: current time
    :param start: start time
    :param i: iteration through this epoch
    :param epoch: epoch number
    :param num_epochs: total no. of epochs
    :param filename: the filename to save
    """
    elap = time - start
    progress = epoch * steps + i + 1
    rem = num_epochs * steps - progress
    ETA = rem / progress * elap
    hrs = int(ETA / 3600)
    minutes = int((ETA / 3600 % 1) * 60)
    # save_res = np.array([epoch, num_epochs, i, steps, hrs, minutes])
    # np.save(progress_dir + filename, save_res)
    print('[%d/%d][%d/%d]\tETA: %d hrs %d mins'
          % (epoch, num_epochs, i, steps,
             hrs, minutes))