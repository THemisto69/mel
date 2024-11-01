import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint

torch.backends.cudnn.benchmark = True

def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device(f'cuda:{rank}')

    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("Checkpoints directory:", a.checkpoint_path)

    # Checkpoint loading
    cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
    cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    last_epoch = -1
    if cp_g and cp_do:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if cp_g and cp_do:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(a)

    trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels, h.hop_size, h.win_size, h.sampling_rate, 
                          h.fmin, h.fmax, shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device)
    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None
    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False, sampler=train_sampler, 
                              batch_size=h.batch_size, pin_memory=True, drop_last=True)

    if rank == 0:
        validset = MelDataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels, h.hop_size, h.win_size, 
                              h.sampling_rate, h.fmin, h.fmax, shuffle=False, fmax_loss=h.fmax_for_loss, device=device)
        validation_loader = DataLoader(validset, num_workers=1, batch_size=1, pin_memory=True, drop_last=True)
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    mpd.train()
    msd.train()

    for epoch in range(max(0, last_epoch), a.training_epochs):
        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            x, y, _, y_mel = batch
            x, y, y_mel = x.to(device), y.to(device), y_mel.to(device)
            y = y.unsqueeze(1)

            y_g_hat = generator(x)
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                          h.fmin, h.fmax_for_loss)

            optim_d.zero_grad()
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f = discriminator_loss(y_df_hat_r, y_df_hat_g)[0]
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s = discriminator_loss(y_ds_hat_r, y_ds_hat_g)[0]
            (loss_disc_f + loss_disc_s).backward()
            optim_d.step()

            optim_g.zero_grad()
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f, loss_fm_s = feature_loss(fmap_f_r, fmap_f_g), feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_all = sum(generator_loss(y_df_hat_g)) + sum(generator_loss(y_ds_hat_g)) + loss_fm_f + loss_fm_s + loss_mel
            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                if steps % a.checkpoint_interval == 0:
                    save_checkpoint(os.path.join(a.checkpoint_path, f'g_{steps}.pth'), {'generator': generator.state_dict()})
                    save_checkpoint(os.path.join(a.checkpoint_path, f'do_{steps}.pth'), {'mpd': mpd.state_dict(), 'msd': msd.state_dict(), 
                                    'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps, 'epoch': epoch})

                if steps % a.validation_interval == 0:
                    generator.eval()
                    val_err_tot = sum(F.l1_loss(y_mel.to(device), mel_spectrogram(generator(x.to(device)).squeeze(1), h.n_fft, h.num_mels, 
                              h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)).item() for x, y, _, y_mel in validation_loader) / len(validation_loader)
                    sw.add_scalar("validation/mel_spec_error", val_err_tot, steps)
                    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
        if epoch == a.training_epochs - 1 and rank == 0:
            torch.save(generator.state_dict(), os.path.join(a.checkpoint_path, 'final_model.pth'))
            print('Training complete and final model saved.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='/content/drive/MyDrive/hifigan_checkpoints')
    parser.add_argument('--config', default='/content/drive/MyDrive/config_v1.json')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    a = parser.parse_args()

    with open(a.config) as f:
        h = AttrDict(json.load(f))
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        h.num_gpus = torch.cuda.device_count()
        h.batch_size //= h.num_gpus
    else:
        h.num_gpus = 0

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h))
    else:
        train(0, a, h)

if __name__ == '__main__':
    main()
