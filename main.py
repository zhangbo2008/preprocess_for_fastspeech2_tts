#========研究一下这个预处理代码

# 首先我们下载好我们的LJ数据集:
# https://keithito.com/LJ-Speech-Dataset/  才2G多.
#首先我们了解数据集:
'''

一个wav文件夹里面装的都是纯音频
一个meatadata里面是一个csv
csv的文件说明:
FILE FORMAT

Metadata is provided in metadata.csv. This file consists of one record per
line, delimited by the pipe character (0x7c). The fields are:

  1. ID: this is the name of the corresponding .wav file
  2. Transcription: words spoken by the reader (UTF-8)
  3. Normalized Transcription: transcription with numbers, ordinals, and
     monetary units expanded into full words (UTF-8).

具体就是这样
LJ003-0223|The roof of the female prison, says the grand jury in their presentment in 1813, let in the rain.|The roof of the female prison, says the grand jury in their presentment in eighteen thirteen, let in the rain.

第一个是wav的文件名, 第二个是文本, 第三个是归一化之后的文本. 也就是真正朗读的内容.
'''





import os
import glob
import tqdm
import torch
import argparse
import numpy as np
from utils.stft import TacotronSTFT
from utils.util import read_wav_np
from dataset.audio_processing import pitch
from utils.hparams import HParam


def main(args, hp):
    stft = TacotronSTFT(
        filter_length=hp.audio.n_fft,
        hop_length=hp.audio.hop_length,
        win_length=hp.audio.win_length,
        n_mel_channels=hp.audio.n_mels,
        sampling_rate=hp.audio.sample_rate,
        mel_fmin=hp.audio.fmin,
        mel_fmax=hp.audio.fmax,
    )

    wav_files = glob.glob(os.path.join(args.data_path, "**", "*.wav"), recursive=True)
    mel_path = os.path.join(hp.data.data_dir, "mels")
    energy_path = os.path.join(hp.data.data_dir, "energy")
    pitch_path = os.path.join(hp.data.data_dir, "pitch")
    os.makedirs(mel_path, exist_ok=True)
    os.makedirs(energy_path, exist_ok=True)
    os.makedirs(pitch_path, exist_ok=True)
    print("Sample Rate : ", hp.audio.sample_rate)
    for wavpath in tqdm.tqdm(wav_files, desc="preprocess wav to mel"):
        sr, wav = read_wav_np(wavpath, hp.audio.sample_rate)
        p = pitch(wav, hp)  # [T, ] T = Number of frames  # p算的是f0
        wav = torch.from_numpy(wav).unsqueeze(0)
        mel, mag = stft.mel_spectrogram(wav)  # mel [1, 80, T]  mag [1, num_mag, T]
        mel = mel.squeeze(0)  # [num_mel, T]
        mag = mag.squeeze(0)  # [num_mag, T]
        e = torch.norm(mag, dim=0)  # [T, ]  #计算2范数,得到能量.
        p = p[: mel.shape[1]]
        id = os.path.basename(wavpath).split(".")[0]
        np.save("{}/{}.npy".format(mel_path, id), mel.numpy(), allow_pickle=False)
        np.save("{}/{}.npy".format(energy_path, id), e.numpy(), allow_pickle=False)
        np.save("{}/{}.npy".format(pitch_path, id), p, allow_pickle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_path", type=str, required=False, help="root directory of wav files"
    )
    parser.add_argument(
        "-c", "--config", type=str, required=False, help="yaml file for configuration"
    )
    args = parser.parse_args()

    args.config = 'configs/default.yaml'
    args.data_path = 'LJ_sample\wavs'

    hp = HParam(args.config)

    main(args, hp)
