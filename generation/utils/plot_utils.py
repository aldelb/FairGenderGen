import numpy as np
import pandas as pd
from os.path import join
import constants.constants as constants
import matplotlib
from matplotlib import pyplot as plt

def plotHistEpoch(file):
    file_path = join(constants.saved_path, file)
    metrics_df = pd.read_csv(file_path, delimiter=";")
    epoch = metrics_df["epoch"].values
    g_loss = metrics_df["g_loss"].values
    val_g_loss = metrics_df["val_g_loss"].values
    loss_eye = metrics_df["loss_eye"].values
    val_loss_eye = metrics_df["val_loss_eye"].values
    loss_pose_r = metrics_df["loss_pose_r"].values
    val_loss_pose_r = metrics_df["val_loss_pose_r"].values
    loss_au = metrics_df["loss_au"].values
    val_loss_au = metrics_df["val_loss_au"].values
    d_loss = metrics_df["d_loss"].values
    real_pred = metrics_df["real_pred"].values
    fake_pred = metrics_df["fake_pred"].values
    basic_real_pred = metrics_df["basic_real_pred"].values
    basic_fake_pred = metrics_df["basic_fake_pred"].values
    plotHistPredEpochGAN(epoch, real_pred, fake_pred, basic_real_pred, basic_fake_pred)
    plotHistLossEpoch(epoch, g_loss, val_g_loss, d_loss)
    plotHistAllLossEpoch(epoch, loss_eye, val_loss_eye, loss_pose_r, val_loss_pose_r, loss_au, val_loss_au)
    

def plotHistLossEpoch(epoch, g_loss, val_g_loss, d_loss):
    plt.figure(dpi=100)
    plt.plot(epoch, g_loss, label='Generator loss')
    plt.plot(epoch, g_loss, label='Generator loss', color="blue")
    plt.plot(epoch, val_g_loss, label='Val_gen loss', color="lightblue")
    plt.plot(epoch, d_loss, label='d_loss', color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(constants.saved_path+f'loss_epoch_{epoch[-1]}.png')
    plt.close()

def plotHistAllLossEpoch(epoch, loss_eye, val_loss_eye, loss_pose_r, val_loss_pose_r, loss_au, val_loss_au):
    plt.figure(dpi=100)
    plt.plot(epoch, loss_eye, color="darkgreen", label='loss_eye')
    plt.plot(epoch, val_loss_eye, color="limegreen", label='val_loss_eye')

    plt.plot(epoch, loss_pose_r, color="darkblue", label='loss_pose_r')
    plt.plot(epoch, val_loss_pose_r, color="cornflowerblue", label='val_loss_pose_r')

    plt.plot(epoch, loss_au, color="red", label='loss_au')
    plt.plot(epoch, val_loss_au, color="lightcoral", label='val_loss_au')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(constants.saved_path+f'all_loss_epoch_{epoch[-1]}.png')
    plt.close()


def plotHistPredEpochGAN(epoch, real_pred, fake_pred, basic_real_pred, basic_fake_pred):
    plt.figure(dpi=100)

    plt.plot(epoch, real_pred, color="green", label='Real')
    plt.plot(epoch, basic_real_pred, color="lightgreen", label='Basic Real')
    plt.plot(epoch, fake_pred, color="red", label='Fake')
    plt.plot(epoch, basic_fake_pred, color="lightcoral", label='Basic Fake')

    plt.yticks(np.arange(0, 1, step=0.2)) 
    plt.xlabel("Epoch")
    plt.ylabel("Discriminator prediction")
    plt.legend()
    plt.savefig(constants.saved_path+f'pred_epoch_{epoch[-1]}.png')
    plt.close()