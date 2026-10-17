import json
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from data.ecg_augment import ECGAugment
from scipy.signal import stft
from skimage.transform import resize
import pywt
import matplotlib
from scipy import signal


class AFDataset(Dataset): # PyTorch Dataset
    def __init__(self, path1, path2, split='train', fold=1, transform=None, oversample=False, ga_test=False):
        self.path1 = path1
        self.path2 = path2
        json_path = os.path.join(path1,'fold_'+str(fold)+'.json') # Sets the path name to the JSON file for the specific fold provided to the function. 
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f) # Loads the JSON to data
        if ga_test == True: # If ga_test is True, the file names are set to the two subsets of data. 
            self.file_names = data['train'] + data['val']
        else: # If ga_test is False, the file name is set to that of the subset provided by the split variable. 
            self.file_names = data[split]
        self.transform = transform 
        self.oversample = oversample and (split == 'train')

        if 'cpsc2021' in path1: # For CPSC2021 dataset
            self.X, self.y = self._load_all_segments_cpsc2021() # Defined below
        else: # For AFDB dataset
            self.X, self.y = self._load_all_segments() # Defined below

        # Set the labels to 0 or 1
        self.y = np.array([1 if label == 'AFIB' else 0 for label in self.y])

        # If oversample is True, oversample the minority class in order to compensate for class imbalance. 
        if self.oversample:
            self.X, self.y = self._oversample_minority_class() # Defined below

    # Each .npz file's data['segments'] has the format: (n_segments, L, C) (the number of segments, the number of samples per segment, and the leads)

    def _load_all_segments_cpsc2021(self): # Function to load data into self.X and self.y for CPSC2021
        all_X = [] # Initialise the segment list
        all_y = [] # Initialise the label list
        json_path = os.path.join(self.path1,'case_to_files.json') # Sets the JSON file path
        with open(json_path, 'r', encoding='utf-8') as f: # Loads the JSON data
            json_data = json.load(f)
        for fname in self.file_names: # self.file names is either data['train'] or data['val']] (or data['test'])
            for fcase in json_data[fname]: # For each case in the train, val, or test data subset
                npz_path = os.path.join(self.path2, fcase) # Path to the case
                data = np.load(npz_path) # Loads the case
                X = data['segments'] # Loads the number of segments from the case
                y = data['labels'] # 1D array containing 1/0 labels for the segments
                if X.ndim < 3: 
                    continue
                all_X.append(X) # Append the segments of this case to the list of all segments
                all_y.append(y) # Append the labels of this case to the list of all labels
        X_full = np.concatenate(all_X, axis=0) # Concatenates all of the cases, summing the number of segments in the first column
        y_full = np.concatenate(all_y, axis=0) # Concatenates all the labels

        # For the X data, the concatenate does as follows: 
            # case A: (120, L, C)
            # case B: ( 85, L, C)   →   X_full: (120 + 85 + 40, L, C) = (245, L, C)
            # case C: ( 40, L, C)

        # y_full[i] is the label for X_full[i]
    
        return X_full, y_full

    def _load_all_segments(self): # Function to load data into self.X and self.y for AFDB
        all_X = [] # Initalise the list of segments
        all_y = [] # Intialise the list of labels
        for fname in self.file_names: # self.file names is either data['train'] or data['val']] (or data['test'])
            npz_path = os.path.join(self.path2, fname+'.npz') # Sets the path to the .npz file (contains ECG data)
            data = np.load(npz_path) # Loads the data
            X = data['segments'] 
            y = data['labels']
            if X.ndim < 3:
                continue
            all_X.append(X) # Append the segment data to all_X
            all_y.append(y) # Append the label data to all_y
        X_full = np.concatenate(all_X, axis=0)
        y_full = np.concatenate(all_y, axis=0)

        # y_full[i] is the label for X_full[i]

        return X_full, y_full

    def _generate_image(self, seg): # Spectrogram image generation function

        fs = 250 # Sampling frequency
        signal = seg[:, 0].astype(np.float32) # Take only the segment data from the data vector
        # STFT
        win_sec = 1
        nperseg = int(win_sec * fs) # 250
        noverlap = nperseg // 2 # 125 (Nyquist Sampling Theorem)
        f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap) # STFT
        mag = np.abs(Zxx)  # (freq_bins, time_frames) 

        # 0–40 Hz）
        f_max_plot = 40.0 
        freq_mask = f <= f_max_plot
        mag = mag[freq_mask, :] # No Butterworth filter applied - just takes the frequencies in a certain range

        # dB
        mag = 20 * np.log10(mag + 1e-8) # Converts to dB
        mag = np.clip(mag, -80, mag.max()) # Clips the dB range between -80 and mag.max()

        # [0, 1]
        mag = (mag + 80) / 100.0 # Normalises the dB values to 0=1 range 

        # (128, 128)
        mag_resized = resize(mag, (128,128), anti_aliasing=True, preserve_range=True) # Converts the vector to a matrix of 128x128 pixels

        #  matplotlib colormap - RGB
        cmap = matplotlib.colormaps.get_cmap('jet') # Greyscale to RGB
        rgb = cmap(mag_resized)[:, :, :3]  # alpha 

        return rgb

    def _oversample_minority_class(self): # Function to deal with class imbalance 
        """
        (AFIB, label=1)
        (N, label=0)
        """
        X = self.X 
        y = self.y

        # Finds the indices of each class in the label array
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]

        # Sections the data into each class
        X_0, y_0 = X[idx_0], y[idx_0]
        X_1, y_1 = X[idx_1], y[idx_1]

        # AF is the minority (fewer segments)
        n_majority = len(X_0)
        n_minority = len(X_1)

        if n_minority >= n_majority:
            print(f"[Oversample] No need: minority ({n_minority}) >= majority ({n_majority})")
            return X, y

        np.random.seed(42)
        oversample_idx = np.random.choice(idx_1, size=n_majority - n_minority, replace=True) # Choose some AF segment indices
        X_1_oversampled = X[oversample_idx] 
        y_1_oversampled = y[oversample_idx]

        # Adds the oversampled (double-sampled essentially) samples to the dataset
        X_balanced = np.concatenate([X_0, X_1, X_1_oversampled], axis=0)
        y_balanced = np.concatenate([y_0, y_1, y_1_oversampled], axis=0)

        print(f"[Oversample] Done: {len(X_0)} normal + {len(X_1) + len(X_1_oversampled)} AF (original: {len(X_1)})")

        return X_balanced, y_balanced

    def _add_baseline_wander(self, x, fs=250, amplitude_ratio=0.7): # Add baseline wander to the data
        """
        Add baseline wander to multi-lead ECG signal of shape (L, C).
        
        Args:
            x: ECG signal, shape (L,) or (L, C)
            fs: Sampling frequency in Hz
            amplitude_ratio: Ratio of wander amplitude to global peak-to-peak
        
        Returns:
            Noisy signal with same shape as input
        """
        original_shape = x.shape

        # Handle single-lead case: (L,) → (L, 1)
        if x.ndim == 1:
            x = x[:, np.newaxis]  # (L, 1)
            single_lead = True
        else:
            assert x.ndim == 2, "Input must be (L,) or (L, C)"
            single_lead = False

        L, C = x.shape

        # Compute global peak-to-peak across all leads and time
        global_ptp = np.max(x) - np.min(x)
        amp = global_ptp * amplitude_ratio

        # Randomly sample frequency in physiological range
        # freq = np.random.uniform(0.1, 0.5)  # Hz
        freq = 0.1
        phase = np.random.uniform(0, 2 * np.pi)

        # Time vector
        t = np.arange(L) / fs  # shape: (L,)

        # Generate baseline wander (same for all leads)
        wander = amp * np.sin(2 * np.pi * freq * t + phase)  # shape: (L,)

        # Broadcast to all channels: (L,) + (L, C) → (L, C)
        x_noisy = x + wander[:, np.newaxis]  # explicitly make (L, 1) for broadcasting

        # Restore original shape
        if single_lead:
            x_noisy = x_noisy.squeeze(axis=1)  # back to (L,)

        return x_noisy

    def _add_powerline_noise(self, x, fs=250, f0=50, snr_db=20, harmonics=2): # Adds powerline noise
        """
        Add powerline noise to ECG signal of shape (L,) or (L, C).
        All channels share the same interference (physiologically realistic).
        """
        original_shape = x.shape

        # Ensure at least 2D for unified processing: (L, C)
        if x.ndim == 1:
            x = x[:, np.newaxis]  # (L, 1)
            single_lead = True
        else:
            assert x.ndim == 2, "Input must be (L,) or (L, C)"
            single_lead = False

        L, C = x.shape
        t = np.arange(L) / fs  # (L,)

        # Generate noise (same for all leads)
        noise = np.zeros(L)
        for h in range(1, harmonics + 1):
            phase = np.random.uniform(0, 2 * np.pi)
            noise += (1.0 / h) * np.sin(2 * np.pi * h * f0 * t + phase)

        # Compute powers
        signal_power = np.mean(x ** 2)      # scalar, over all elements
        noise_power = np.mean(noise ** 2)   # scalar

        if noise_power == 0:
            return x.reshape(original_shape)

        # Scale to achieve target SNR
        desired_noise_power = signal_power / (10 ** (snr_db / 10))
        scale = np.sqrt(desired_noise_power / noise_power)

        # Add noise (broadcast to all channels)
        x_noisy = x + scale * noise[:, np.newaxis]  # (L, C) + (L, 1)

        # Restore original shape
        if single_lead:
            x_noisy = x_noisy.squeeze(axis=1)
        return x_noisy

    def _add_emg_noise(self, x, snr_db=0.5, bandwidth=(20, 500), fs=250):
        original_shape = x.shape
        if x.ndim == 1:
            x = x[:, np.newaxis]
            single_lead = True
        else:
            assert x.ndim == 2, "Expected (L,) or (L, C)"
            single_lead = False

        L, C = x.shape
        signal_power = np.mean(x ** 2)

        # Nyquist Theorem
        nyq = 0.5 * fs
        low, high = bandwidth
        low = max(low, 1)         
        high = min(high, nyq * 0.999) 
        
        if low >= high:
            noise = np.random.randn(L, C)
        else:
            noise = np.random.randn(L, C)
            b, a = signal.butter(4, [low / nyq, high / nyq], btype='band')
            for c in range(C):
                noise[:, c] = signal.filtfilt(b, a, noise[:, c])

        noise_power = np.mean(noise ** 2)
        if noise_power == 0:
            return x.reshape(original_shape)

        scale = np.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10))))
        x_noisy = x + scale * noise

        if single_lead:
            x_noisy = x_noisy.squeeze(axis=1)
        return x_noisy

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx): # Builds a single training sample
        x = self.X[idx]
        y = self.y[idx]
        if self.transform: # Apply the transformation if one is provided
            x = self.transform(x)
        # x = self._add_emg_noise(x)
        img = self._generate_image(x) # Build the spectrogram image
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # Reorder the image from (H, W, C) to (C, H, W)
        x = torch.tensor(x, dtype=torch.float32)[:, 0:1]  # Keep only the first lead
        
        y = torch.tensor(y, dtype=torch.long)
        return x, img, y
