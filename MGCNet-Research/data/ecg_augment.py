import numpy as np

class ECGAugment:

    def __init__(self, # Parameters for data augmentation
                 target_fs=200,  # Target sampling frequency
                 original_fs=200,  # Original sampling frequency
                 prob=0.5, # Adds randomness to the samples which are augmented
                 noise_std=0.1,  
                 time_warp_sigma=10,  
                 time_mask_ratio=0.1,  
                 amp_scale_range=0.4,  
                 zoom_in_ratio_range=0.5,  # time_zoom_in 
                 zoom_out_expand_range=0.5,  # time_zoom_out 
                 norm_type='zscore'):

        self.target_fs = target_fs
        self.original_fs = original_fs
        self.prob = prob
        self.noise_std = noise_std
        self.time_warp_sigma = time_warp_sigma
        self.time_mask_ratio = time_mask_ratio
        self.amp_scale_range = amp_scale_range
        self.zoom_in_ratio_range = zoom_in_ratio_range
        self.zoom_out_expand_range = zoom_out_expand_range
        self.norm_type = norm_type

        self.resample_ratio = target_fs / original_fs

    def __call__(self, x):
        x = x.copy()

        # These functions are defined below
        x = self._resample_to_fixed_fs(x)

        if np.random.rand() < self.prob:
            x = self._amplitude_scaling(x)

        if np.random.rand() < self.prob:
            x = self._vertical_flip(x) 

        if np.random.rand() < self.prob:
            if np.random.rand() < 0.5:
                x = self._time_zoom_in(x)
            else:
                x = self._time_zoom_out(x)

        if np.random.rand() < self.prob:
            x = self._random_time_warp(x)

        if np.random.rand() < self.prob:
            x = self._random_time_mask(x)

        if np.random.rand() < self.prob:
            x = self._add_gaussian_noise(x)

        # Normalization applied to all of the samples
        x = self._normalize(x)

        return x

    def _vertical_flip(self, x): # I don't think this is likely to be very useful for augmenting data for a real use case. 
        return -x

    def _resample_to_fixed_fs(self, x):
        seq_len = x.shape[0]

        # Scale the length of the sequence by the resample_ratio (higher target fs, more samples)
        new_len = int(seq_len * self.resample_ratio)
        new_len = max(10, new_len)

        # The spacing of new samples
        t_old = np.linspace(0, seq_len - 1, seq_len)
        t_new = np.linspace(0, seq_len - 1, new_len)

        if x.ndim == 1: # Interpolates the new samples from the original data x. 
            x_resampled = np.interp(t_new, t_old, x)
        else:
            x_resampled = np.array([np.interp(t_new, t_old, x[:, c]) for c in range(x.shape[1])]).T

        return x_resampled

    def _add_gaussian_noise(self, x):
        signal_std = np.std(x)
        noise = np.random.normal(0, self.noise_std * signal_std, x.shape)
        return x + noise

    def _random_time_warp(self, x):
        sigma = self.time_warp_sigma
        seq_len = x.shape[0]
        dx = np.random.randn(seq_len) * sigma

        if seq_len > 5:
            dx = np.convolve(dx, np.hanning(5), 'same')

        t = np.arange(seq_len).astype(float) + dx
        t = np.clip(t, 0, seq_len - 1)

        if x.ndim == 1:
            x = np.interp(t, np.arange(seq_len), x)
        else:
            for c in range(x.shape[1]):
                x[:, c] = np.interp(t, np.arange(seq_len), x[:, c])
        return x

    def _random_time_mask(self, x): # Blanks out the signal for a masked period
        ratio = np.random.uniform(0.01, self.time_mask_ratio)
        duration = int(ratio * x.shape[0])
        duration = max(1, duration)
        start = np.random.randint(0, x.shape[0] - duration + 1)
        x[start:start + duration] = 0
        return x

    def _amplitude_scaling(self, x):
        scale = np.random.uniform(1 - self.amp_scale_range, 1 + self.amp_scale_range)
        return x * scale

    def _time_zoom_in(self, x):
        seq_len = x.shape[0]

        crop_ratio = np.random.uniform(1 - self.zoom_in_ratio_range, 1)
        crop_len = int(seq_len * crop_ratio)

        start = np.random.randint(0, seq_len - crop_len + 1)

        if x.ndim == 1:
            x_cropped = x[start:start + crop_len]
        else:
            x_cropped = x[start:start + crop_len, :]

        t_crop = np.linspace(0, crop_len - 1, crop_len)
        t_old = np.linspace(0, crop_len - 1, seq_len)  # 目标长度为原长度

        if x.ndim == 1:
            x_zoomed = np.interp(t_old, t_crop, x_cropped)
        else:
            x_zoomed = np.array([np.interp(t_old, t_crop, x_cropped[:, c]) for c in range(x_cropped.shape[1])]).T

        return x_zoomed

    def _time_zoom_out(self, x):
        seq_len = x.shape[0]

        expand_ratio = np.random.uniform(1, 1 + self.zoom_out_expand_range)
        expand_len = int(seq_len * expand_ratio)

        pad_len = expand_len - seq_len
        pad_left = pad_len // 2
        pad_right = pad_len - pad_left

        if x.ndim == 1:
            x_padded = np.pad(x, (pad_left, pad_right), mode='constant')
        else:
            x_padded = np.pad(x, ((pad_left, pad_right), (0, 0)), mode='constant')

        t_padded = np.linspace(0, expand_len - 1, expand_len)
        t_old = np.linspace(0, expand_len - 1, seq_len)

        if x.ndim == 1:
            x_zoomed = np.interp(t_old, t_padded, x_padded)
        else:
            x_zoomed = np.array([np.interp(t_old, t_padded, x_padded[:, c]) for c in range(x_padded.shape[1])]).T

        return x_zoomed

    def _normalize(self, x):
        if self.norm_type is None:
            return x

        if self.norm_type == 'zscore':
            mean = np.mean(x, axis=0, keepdims=True)
            std = np.std(x, axis=0, keepdims=True)
            std[std < 1e-6] = 1e-6
            return (x - mean) / std

        elif self.norm_type == 'minmax':
            min_val = np.min(x, axis=0, keepdims=True)
            max_val = np.max(x, axis=0, keepdims=True)
            gap = (max_val - min_val)
            gap[gap < 1e-6] = 1e-6
            return (x - min_val) / gap

        else:
            raise ValueError(f"Error: {self.norm_type}")