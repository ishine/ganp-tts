import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import itertools

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from model import ScheduledOptim

from model.hifigan import  MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, discriminator_loss, generator_loss

from model.srgan  import  Discriminator

from dataset import Dataset

from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print("Prepare training ...")
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    des_w = train_config["optimizer"]["des_w"]
    ges_w = train_config["optimizer"]["ges_w"]
    gan_start = int(train_config["optimizer"]["gan_start"])

    print(train_config)
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        num_workers=16,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    #model_des = Discriminator().to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

   

    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                0.0002, betas=[0.8, 0.99])
    #optim_g = torch.optim.AdamW(model.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

   
    if    args.restore_step > 10:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        print("restore mpd msd start")
        mpd.load_state_dict(ckpt["desmodel"])            
        msd.load_state_dict(ckpt["desmodel1"])            
        optim_d.load_state_dict(ckpt['desoptimizer'])
        print("restore mpd msd end")

    model = nn.DataParallel(model)
    model_des = nn.DataParallel(mpd)
    model_des1 = nn.DataParallel(msd)


    num_param = get_param_num(model)
    num_param1 = get_param_num(model_des)
    num_param2 = get_param_num(model_des1)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param,num_param1,num_param2)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    mse_loss = nn.MSELoss().to(device)
    mae_loss = nn.L1Loss().to(device)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()
    model_des1.train()
    model_des.train()
    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                if 1:

                # Forward
                  output = model(*(batch[2:]))
                  (
                    mel_predictions,
                    postnet_output,
                    p_predictions,
                    e_predictions,
                    log_d_predictions,
                    d_rounded,
                    src_masks,
                    mel_masks,
                    src_lens,
                    mel_lens,
                  )  = output
                  
                # Cal Loss
                  mel_targets = batch[6]
                  if mel_targets.shape[1] != postnet_output.shape[1]:
                      mel_targets1 = torch.split(mel_targets,postnet_output.shape[1],dim=1)
                      mel_targets = mel_targets1[0]
                #mel_predictions = output[0]
                  optim_d.zero_grad()
                   # MPD
                  y = torch.unsqueeze(mel_targets,1)
                  y_g_hat = torch.unsqueeze(postnet_output,1)
                  #print("mel_target",mel_targets.shape)
                  #print("postnet_output",postnet_output.shape)
                  y_df_hat_r, y_df_hat_g, _, _ = model_des(y, y_g_hat.detach())
                  loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

                   # MSD
                  y_ds_hat_r, y_ds_hat_g, _, _ = model_des1(y, y_g_hat.detach())
                  loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                  loss_disc_all = loss_disc_s + loss_disc_f


                  loss_disc_all.backward()
                  optim_d.step()




                  # Generator
                  optimizer.zero_grad()
                  #optim_g.zero_grad()
                  loss_mel = Loss(batch, output)
                  losses = loss_mel
                   # L1 Mel-Spectrogram Loss
#                  loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

                  y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
                  y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
                  loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                  loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                  loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                  loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                  if step > gan_start:  
                       loss_gen_all = ges_w * (loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f) + loss_mel[0]
                  else:
                       loss_gen_all = loss_mel[0]

                  loss_gen_all.backward()
                  optimizer.step_and_update_lr()
                   #optim_g.step()

                  #total_loss = losses[0] + ges_w * (ges_loss + logit1_loss + logit2_loss + logit3_loss)

                # Backward
                #total_loss = total_loss 

               # total_loss.backward()

                if step % grad_acc_step == 0:
                    pass
                    # Clipping gradients to avoid gradient explosion
                    #nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    #nn.utils.clip_grad_norm_(model_des.parameters(), grad_clip_thresh)
                    #des_optim.step_and_update_lr()
                    #des_optim.zero_grad()

                    # Update weights
                    #optimizer.step_and_update_lr()
                    #optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    train_logger.add_scalar("Loss/des", loss_disc_all, step)
                    train_logger.add_scalar("Loss/loss_gen_s", loss_gen_s, step)
                    train_logger.add_scalar("Loss/loss_gen_f", loss_gen_f, step)
                    train_logger.add_scalar("Loss/loss_fm_s", loss_fm_s, step)
                    train_logger.add_scalar("Loss/loss_fm_f", loss_fm_f, step)
                    #train_logger.add_scalar("Loss/logit", logit_loss, step)
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                        *losses
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses)

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(model, step, configs, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                            "desmodel": model_des.module.state_dict(),
                            "desmodel1": model_des1.module.state_dict(),
                            "desoptimizer": optim_d.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
