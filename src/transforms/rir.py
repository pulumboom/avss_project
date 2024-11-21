import torch
from torch import nn
import wget
from zipfile import ZipFile
import os
import glob
import numpy as np
import torchaudio
import subprocess

# inspired by https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tutorials/asr-noise-augmentation-offline.html


class AddRir(nn.Module):

    def __init__(
            self,
            apply_prob=0.5,
            pointsource_noises=False,
            real_rirs_isotropic_noises=False,
            simulated_rirs=True
        ):
        super().__init__()

        self.apply_prob = apply_prob

        noise_samples = 'noise_samples'
        if not os.path.exists(noise_samples):
            os.makedirs(noise_samples)
        if not os.path.exists('noise_samples/rirs_noises.zip'):
            rirs_noises_url = 'https://www.openslr.org/resources/28/rirs_noises.zip'  
            rirs_noises_path = wget.download(rirs_noises_url, noise_samples)
            print(f"Noise dataset downloaded at: {rirs_noises_path}")
        else:
            print("Zipfile already exists.")
            rirs_noises_path = 'noise_samples/rirs_noises.zip'

        if not os.path.exists('noise_samples/RIRS_NOISES'):
            # try:
            with ZipFile(rirs_noises_path, "r") as zipObj:
                zipObj.extractall(noise_samples)
                print("Extracting noise data complete")
            # Convert 8-channel audio files to mono-channel
            wav_list = glob.glob(noise_samples + '/RIRS_NOISES/**/*.wav', recursive=True)
            for wav_path in wav_list:
                mono_wav_path = wav_path[:-4] + '_mono.wav'
                cmd = f"sox {wav_path} {mono_wav_path} remix 1"
                subprocess.call(cmd, shell=True)
            print("Finished converting the 8-channel noise data .wav files to mono-channel")
            # except Exception:
            #     print("Not extracting. Extracted noise data might already exist.")
        else: 
            print("Extracted noise data already exists. Proceed to the next step.")

        self.noise_path = []

        if pointsource_noises:
            self._get_wav('pointsource_noises')

        if real_rirs_isotropic_noises:
            self._get_wav('real_rirs_isotropic_noises')

        if simulated_rirs:
            self._get_wav('simulated_rirs')


    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            x (Tensor): normalized tensor.
        """
        if torch.rand(1) > self.apply_prob:
            return x
        
        snr = np.random.choice([5., 10., 15.])
        noise_f = np.random.choice(self.noise_path)
        noise_t, _ = torchaudio.load(noise_f)

        if noise_t.shape[-1] > x.shape[-1]:
            noise_t = noise_t[..., :x.shape[-1]]

        noise_t = torch.as_tensor(self._concatenate_noise_sample(noise_t, 32000)).float()

        noisy_x = self._snr_mixer(x, noise_t, snr)
        return noisy_x


    def _get_wav(self, name):
        for dirpath, _, files in os.walk(f'noise_samples/RIRS_NOISES/{name}'):
            for file in files:
                if file[-9:] != '_mono.wav':
                    continue
                
                self.noise_path.append(dirpath + '/' + file)

    def _concatenate_noise_sample(self, noise, len_clean):
        while len(noise) <= len_clean:
            noiseconcat = np.append(noise, np.zeros(16000))
            noise = np.append(noiseconcat, noise)

        if noise.size > len_clean:
            noise = noise[0:len_clean]

        return noise


    def _snr_mixer(self, clean, noise, snr):
        # Normalizing to -25 dB FS
        rmsclean = (clean**2).mean()**0.5
        if rmsclean == 0:
            rmsclean = 1
        
        scalarclean = 10 ** (-25 / 20) / rmsclean
        clean = clean * scalarclean
        rmsclean = (clean**2).mean()**0.5

        rmsnoise = (noise**2).mean()**0.5
        if rmsnoise == 0:
            rmsnoise = 1
        
        scalarnoise = 10 ** (-25 / 20) /rmsnoise
        noise = noise * scalarnoise
        rmsnoise = (noise**2).mean()**0.5
        if rmsnoise == 0:
            rmsnoise = 1
        
        # Set the noise level for a given SNR
        noisescalar = torch.sqrt(rmsclean / (10**(snr/20)) / rmsnoise)
        noisenewlevel = noise * noisescalar
        noisyspeech = clean + noisenewlevel
        return noisyspeech
