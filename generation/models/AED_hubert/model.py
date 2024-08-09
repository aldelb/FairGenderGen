import argparse
from os.path import join
import csv
from datetime import datetime
from math import floor
import torch.nn as nn
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import random

import constants.constants as constants
from utils.data_utils import format_data, reshape_output, concat_with_labels
from utils.noise_generator import NoiseGenerator
from utils.params_utils import save_params
from utils.plot_utils import plotHistEpoch
from utils.model_parts import DoubleConv, Down, OutConv, Up, Conv, DownDiscr, ReverseLayerF
from utils.labels import get_other_label, supress_silence_index, get_no_silence_index_from_one_hot
    

class Generator(pl.LightningModule):
    def __init__(self):
        super().__init__()
        bilinear = True
        factor = 2 if bilinear else 1
        self.b1 = 0.5
        self.b2 = 0.999
        self.noise_g = NoiseGenerator()

        ##encode audio
        self.in1_audio = Conv(constants.audio_dim, 256, constants.first_kernel_size)
        self.in2_audio = Conv(256, 128, constants.first_kernel_size)
        self.down1_audio = Down(128, 64, constants.first_kernel_size) 
        self.down2_audio = Down(64, 128, constants.kernel_size)
        self.down3_audio = Down(128, 256, constants.kernel_size)

        ###concat with noise and labels here
        self.down4 = Down(256 + constants.number_of_dim_labels, 512, constants.kernel_size) # 256 for the audio embedding 
        self.down5 = Down(512, 1024, constants.kernel_size)
        self.down6 = Down(1024, 2048 // factor, constants.kernel_size)


        ##Decoder eye
        self.up1_eye = Up(2048, 1024 // factor, constants.kernel_size, bilinear)
        self.up2_eye = Up(1024, 512 // factor, constants.kernel_size, bilinear)
        self.up3_eye = Up(512, 256 // factor, constants.kernel_size, bilinear)
        self.up4_eye = Up(256, 128 // factor, constants.kernel_size, bilinear)
        self.up5_eye = Up(128, 64, constants.kernel_size, bilinear)
        self.outc_eye = OutConv(64, constants.eye_size, constants.kernel_size)

        ##Decoder pose_r
        self.up1_pose_r = Up(2048, 1024 // factor, constants.kernel_size, bilinear)
        self.up2_pose_r = Up(1024, 512 // factor, constants.kernel_size, bilinear)
        self.up3_pose_r = Up(512, 256 // factor, constants.kernel_size, bilinear)
        self.up4_pose_r = Up(256, 128 // factor, constants.kernel_size, bilinear)
        self.up5_pose_r = Up(128, 64, constants.kernel_size, bilinear)
        self.outc_pose_r = OutConv(64, constants.pose_r_size, constants.kernel_size)

        ##Decoder AUs
        self.up1_au = Up(2048, 1024 // factor, constants.kernel_size, bilinear)
        self.up2_au = Up(1024, 512 // factor, constants.kernel_size, bilinear)
        self.up3_au = Up(512, 256 // factor, constants.kernel_size, bilinear)
        self.up4_au = Up(256, 128 // factor, constants.kernel_size, bilinear)
        self.up5_au = Up(128, 64, constants.kernel_size, bilinear)
        self.outc_au = OutConv(64, constants.au_size, constants.kernel_size)

    def forward(self, x_audio, gender):
        in_audio = torch.swapaxes(x_audio, 1, 2)

        x = self.in1_audio(in_audio)
        x = self.in2_audio(x)
        x1 = self.down1_audio(x)
        x2 = self.down2_audio(x1)
        x3 = self.down3_audio(x2)

        ###concat with noise and labels here
        noise = self.noise_g.getNoise(x3, variance=0.1, interval=[-1,1])
        x_audio_noise = torch.add(x3, noise)

        x_audio_noise = concat_with_labels(x_audio_noise, x_audio_noise.shape[2], gender)

        #Encoder (audio + noise part)
        x4 = self.down4(x_audio_noise)
        x5 = self.down5(x4)
        latent_representation = self.down6(x5)

        #Decoder gaze
        x = self.up1_eye(latent_representation, x5)
        x = self.up2_eye(x, x4)
        x = self.up3_eye(x, x3)
        x = self.up4_eye(x, x2)
        x = self.up5_eye(x, x1)
        logits_eye = self.outc_eye(x)
        logits_eye = torch.tanh(logits_eye)

        #Decoder pose_r
        x = self.up1_pose_r(latent_representation, x5)
        x = self.up2_pose_r(x, x4)
        x = self.up3_pose_r(x, x3)
        x = self.up4_pose_r(x, x2)
        x = self.up5_pose_r(x, x1)
        logits_pose_r = self.outc_pose_r(x)
        logits_pose_r = torch.tanh(logits_pose_r)

        #Decoder AUs
        x = self.up1_au(latent_representation, x5)
        x = self.up2_au(x, x4)
        x = self.up3_au(x, x3)
        x = self.up4_au(x, x2)
        x = self.up5_au(x, x1)
        logits_au = self.outc_au(x)
        logits_au = torch.tanh(logits_au)
        
        logits_eye = torch.swapaxes(logits_eye, 1, 2)
        logits_pose_r = torch.swapaxes(logits_pose_r, 1, 2)
        logits_au = torch.swapaxes(logits_au, 1, 2)
        return latent_representation, logits_eye, logits_pose_r, logits_au

class Discriminator(pl.LightningModule):

    def __init__(self):
        super().__init__()

        ##encode audio
        self.conv1_audio = Conv(constants.audio_dim, 512, constants.first_kernel_size)
        self.conv2_audio = Conv(512, 128, constants.kernel_size)
        self.conv3_audio = Conv(128, 64, constants.kernel_size)

        ##encode behaviour
        self.conv1_behaviour = Conv(constants.pose_size + constants.au_size, 32, constants.first_kernel_size)
        self.conv2_behaviour = Conv(32, 64, constants.kernel_size)

        self.conv_concat = Conv(128 + constants.number_of_dim_labels, 64, constants.kernel_size)
        self.fc1 = torch.nn.Linear(64 * floor(constants.seq_len/4), 64)
        self.fc2 = torch.nn.Linear(64, 1)
    

    def forward(self, x_pose, c_audio, gender):
        in_audio = torch.swapaxes(c_audio, 1, 2)
        c = self.conv1_audio(in_audio)
        c = F.max_pool1d(c, kernel_size=2, stride=2)
        c = self.conv2_audio(c)
        c = F.max_pool1d(c, kernel_size=2, stride=2)
        c = self.conv3_audio(c)
        c = F.max_pool1d(c, kernel_size=2, stride=2)

        ###concat with labels here
        x_audio_labels = c
        x_audio_labels = concat_with_labels(x_audio_labels, x_audio_labels.shape[2], gender)

        x = torch.swapaxes(x_pose, 1, 2)
        x = self.conv1_behaviour(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = self.conv2_behaviour(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)

        x = torch.cat([x, x_audio_labels], dim=1)
        x = self.conv_concat(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
    
class GAN(pl.LightningModule):
    def __init__(self, no_speak_examples=None, speak_examples=None):
        super().__init__()
        self.n_critic = 5
        self.c_lambda = 10
        self.criterion = nn.MSELoss()
        self.create_loss()

        if(no_speak_examples!=None):
            self.no_speak_x_audio, self.no_speak_y, self.no_speak_gender = no_speak_examples
            self.speak_x_audio, self.speak_y, self.speak_gender = speak_examples

        self.automatic_optimization = False
        self.generator = Generator()
        self.discriminator = Discriminator()
        save_params(constants.saved_path, self.generator, self.discriminator)

    def forward(self, x_audio, gender):
        return self.generator(x_audio, gender)

    def configure_optimizers(self):
        b1 = 0.9
        b2 = 0.999
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=constants.g_lr, betas=(b1, b2))
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=constants.d_lr, betas=(b1, b2))
        return g_opt, d_opt

    def on_train_epoch_start(self):
        #torch.cuda.empty_cache()
        #print(str(self.device)+ " memory allocated "+ str(torch.cuda.memory_allocated())+ " max memory allocated "+ str(torch.cuda.max_memory_allocated()))
        self.start_epoch = datetime.now()

    
    def generator_step(self, inputs_audio, targets, gender):
            inputs_audio, targets, target_eye, target_pose_r, target_au = format_data(inputs_audio, targets)
    
            latent_representation, gen_eye, gen_pose_r, gen_au = self.generator(inputs_audio, gender)
            fake_targets = torch.cat((gen_eye, gen_pose_r, gen_au), 2)
            
            with torch.no_grad():
                d_fake_pred = self.discriminator(fake_targets, inputs_audio, gender).squeeze()
                d_real_pred = self.discriminator(targets, inputs_audio, gender).squeeze()

            g_iteration_loss = -1. * torch.mean(d_fake_pred)
            loss_eye = self.criterion(gen_eye, target_eye).mean()
            loss_pose_r = self.criterion(gen_pose_r, target_pose_r).mean()
            loss_au = self.criterion(gen_au, target_au).mean()

            # Calcul de la loss totale du générateur
            g_loss = constants.eye_coeff * loss_eye + constants.pose_coeff * loss_pose_r + constants.au_coeff * loss_au 
            + constants.adversarial_coeff * g_iteration_loss

            return latent_representation, g_loss, loss_eye, loss_pose_r, loss_au, torch.mean(d_fake_pred), torch.mean(d_real_pred)
    

    def discriminator_step(self, inputs_audio, targets, gender):
            inputs_audio, targets, _, _, _ = format_data(inputs_audio, targets)
            with torch.no_grad():
                _, output_eye, output_pose_r, output_au = self.generator(inputs_audio, gender)
                fake_targets = torch.cat((output_eye, output_pose_r, output_au), 2)
            
            selected_audio_inputs, selected_gender, selected_real_targets, selected_fake_targets = self.create_fake_targets(inputs_audio, targets, fake_targets, gender) 

            #fake predictions 
            fake_pred = self.discriminator(selected_fake_targets, selected_audio_inputs, selected_gender).squeeze()
            #real predictions 
            real_pred = self.discriminator(selected_real_targets, selected_audio_inputs, selected_gender).squeeze()

            gp = self.compute_gradient_penalty(selected_real_targets, selected_fake_targets, selected_audio_inputs, selected_gender)
            d_loss = torch.mean(fake_pred) - torch.mean(real_pred) +  self.c_lambda * gp
            
            return d_loss, torch.mean(real_pred), torch.mean(fake_pred)
    
    def create_fake_targets(self, inputs_audio, targets, fake_targets, gender):
            ### Discriminator predictions
            # one third generated by the generator, 
            # one third audio speaking + listening behavior, 
            # one third audio listening +  speaking behavior
            nb_designed = int(inputs_audio.shape[0]/3)
            nb_generated = inputs_audio.shape[0] - 2*nb_designed
            range_1 = min(len(self.no_speak_x_audio), len(self.speak_y))
            idx_1 = random.sample(range(range_1), nb_designed)
            range_2 = min(len(self.speak_x_audio)-1, len(self.no_speak_y)-1)
            idx_2 = random.sample(range(range_2), nb_designed)
            idx_3 = random.sample(range(len(inputs_audio)), nb_generated)

            selected_audio_inputs = torch.cat((self.no_speak_x_audio[idx_1].to(inputs_audio), self.speak_x_audio[idx_2].to(inputs_audio), inputs_audio[idx_3]),0)
            new_order = torch.randperm(selected_audio_inputs.size(0))
            selected_audio_inputs = selected_audio_inputs[new_order]
            selected_gender = torch.cat((self.no_speak_gender[idx_1].to(inputs_audio), self.speak_gender[idx_2].to(inputs_audio), gender[idx_3]),0)[new_order]

            selected_real_targets = torch.cat((self.no_speak_y[idx_1].to(inputs_audio), self.speak_y[idx_2].to(inputs_audio), targets[idx_3]),0)[new_order]
            selected_fake_targets = torch.cat((self.speak_y[idx_1].to(inputs_audio), self.no_speak_y[idx_2].to(inputs_audio), fake_targets[idx_3]),0)[new_order]

            return selected_audio_inputs, selected_gender, selected_real_targets, selected_fake_targets
    

    def compute_gradient_penalty(self, real_samples, fake_samples, inputs_audio, gender):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(len(real_samples), 1, 1, requires_grad=True).to(real_samples)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        interpolates = interpolates.to(real_samples)
        d_interpolates = self.discriminator(interpolates, inputs_audio, gender) #real inputs so i give real labels

        gradient = torch.autograd.grad(
            inputs=interpolates,
            outputs=d_interpolates,
            grad_outputs=torch.ones_like(d_interpolates), 
            create_graph=True,
            retain_graph=True,)[0]
        
        gradient = gradient.reshape(len(gradient), -1)
        gradient_norm = gradient.norm(2, dim=1)
        penalty = torch.mean((gradient_norm - 1)**2)
        return penalty


    def training_step(self, batch, batch_idx):
        inputs_audio, targets, gender = batch
        g_opt, d_opt = self.optimizers()

        for _ in range(self.n_critic):
            d_loss, real_pred, fake_pred = self.discriminator_step(inputs_audio, targets, gender)
            self.d_loss.append(d_loss)
            self.real_pred.append(real_pred)
            self.fake_pred.append(fake_pred)

            d_opt.zero_grad()
            self.manual_backward(d_loss)
            d_opt.step()

        _, g_loss, loss_eye, loss_pose_r, loss_au, d_fake_pred, d_real_pred = self.generator_step(inputs_audio, targets, gender)
        
        self.loss_eye.append(loss_eye)
        self.loss_pose_r.append(loss_pose_r)
        self.loss_au.append(loss_au)
        self.g_loss.append(g_loss)
        self.basic_real_pred.append(d_real_pred)
        self.basic_fake_pred.append(d_fake_pred)
        
        g_opt.zero_grad()
        self.manual_backward(g_loss)
        g_opt.step()
            


    def on_train_epoch_end(self):
        @pl.utilities.rank_zero_only
        def _save_losses(dict_loss):
            file = "lossEpoch.csv"
            metrics = {"device": str(self.device),
                "memory allocated": str(torch.cuda.memory_allocated()),
                "epoch": self.current_epoch,
                "duration": dict_loss["diff"].total_seconds(),
                "g_loss": dict_loss["avg_g_loss"].item(),
                "val_g_loss": dict_loss["avg_val_g_loss"].item(),
                "d_loss": dict_loss["avg_d_loss"].item(),
                "real_pred": dict_loss["avg_real_pred"].item(),
                "fake_pred": dict_loss["avg_fake_pred"].item(),
                "basic_real_pred": dict_loss["avg_basic_real_pred"].item(),
                "basic_fake_pred": dict_loss["avg_basic_fake_pred"].item(),
                "loss_eye": dict_loss["avg_loss_eye"].item(),
                "val_loss_eye": dict_loss["avg_val_loss_eye"].item(),
                "loss_pose_r": dict_loss["avg_loss_pose_r"].item(),
                "val_loss_pose_r": dict_loss["avg_val_loss_pose_r"].item(),
                "loss_au": dict_loss["avg_loss_au"].item(),
                "val_loss_au": dict_loss["avg_val_loss_au"].item(),
                }
            
        
            self.log_metrics_to_csv(metrics, file, mode=current_mode)

            if (self.current_epoch % constants.log_interval == 0):
                plotHistEpoch(file)

        if(self.current_epoch == 0):
            current_mode = "w"
        else:
            current_mode = "a"

        dict_loss = {}
        dict_loss["avg_g_loss"] = torch.stack(self.g_loss).mean()
        dict_loss["avg_val_g_loss"] = torch.stack(self.val_g_loss).mean()
        dict_loss["avg_d_loss"] = torch.stack(self.d_loss).mean()
        dict_loss["avg_real_pred"] = torch.stack(self.real_pred).mean()
        dict_loss["avg_fake_pred"] = torch.stack(self.fake_pred).mean()
        dict_loss["avg_basic_real_pred"] = torch.stack(self.basic_real_pred).mean()
        dict_loss["avg_basic_fake_pred"] = torch.stack(self.basic_fake_pred).mean()
        dict_loss["avg_loss_eye"] = torch.stack(self.loss_eye).mean()
        dict_loss["avg_val_loss_eye"] = torch.stack(self.val_loss_eye).mean()
        dict_loss["avg_loss_pose_r"] = torch.stack(self.loss_pose_r).mean()
        dict_loss["avg_val_loss_pose_r"] = torch.stack(self.val_loss_pose_r).mean()
        dict_loss["avg_loss_au"] = torch.stack(self.loss_au).mean()
        dict_loss["avg_val_loss_au"] = torch.stack(self.val_loss_au).mean()
        dict_loss["diff"] = datetime.now() - self.start_epoch

        _save_losses(dict_loss)
        self.clear_loss()


    def validation_step(self, batch, batch_idx):
        inputs_audio, targets, gender = batch
        with torch.no_grad():
            _, val_g_loss, val_loss_eye, val_loss_pose_r, val_loss_au, val_d_fake_pred, val_d_real_pred = self.generator_step(inputs_audio, targets, gender)

        self.val_g_loss.append(val_g_loss)
        self.val_loss_eye.append(val_loss_eye)
        self.val_loss_pose_r.append(val_loss_pose_r)
        self.val_loss_au.append(val_loss_au)
        self.val_basic_real_pred.append(val_d_real_pred)
        self.val_basic_fake_pred.append(val_d_fake_pred)

        return val_g_loss


    def predict_step(self, batch, batch_idx):
        inputs_audio, details_time, key, gender = batch
        with torch.no_grad():
            latent_representation, output_eye, output_pose_r, output_au = self(inputs_audio.squeeze(1), gender)
        pred = reshape_output(output_eye, output_pose_r, output_au, self.pose_scaler)

        return key, pred, details_time, latent_representation, gender



    ################ Loss processing #########################
    def create_loss(self):
        self.loss_eye = []
        self.loss_pose_r = []
        self.loss_au = []

        self.g_loss = []
        self.d_loss = []
        self.fake_pred = []
        self.real_pred = []    
        self.basic_real_pred = []
        self.basic_fake_pred = []   
        
        self.val_g_loss = []
        self.val_loss_eye = []
        self.val_loss_pose_r = []
        self.val_loss_au = []
        self.val_basic_real_pred = []
        self.val_basic_fake_pred = []

    def clear_loss(self):
        self.loss_eye.clear()
        self.loss_pose_r.clear()
        self.loss_au.clear()

        self.val_g_loss.clear()
        self.val_loss_eye.clear()
        self.val_loss_pose_r.clear()
        self.val_loss_au.clear()
        self.val_basic_real_pred.clear()
        self.val_basic_fake_pred.clear()

        self.g_loss.clear()
        self.d_loss.clear()
        self.fake_pred.clear()
        self.real_pred.clear()
        self.basic_real_pred.clear()
        self.basic_fake_pred.clear()
    
    def log_metrics_to_csv(self, metrics, file, mode="w"):
        # Log metrics to a CSV file
        file_path = join(constants.saved_path, file)

        try:
            with open(file_path, mode, newline="") as file:
                writer = csv.DictWriter(file, fieldnames=metrics.keys(), delimiter=";")
                if mode == "w":
                    writer.writeheader()

                writer.writerow(metrics)
        except Exception as e:
            print(f"Error writing to CSV file: {e}")
    
