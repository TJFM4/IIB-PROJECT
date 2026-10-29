import os
import argparse
import yaml
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ----------------------------
# 导入模块
# ----------------------------
from model.bgm import GMCNet
from data.dataset import AFDataset
from data.ecg_augment import ECGAugment


# ----------------------------
# Grad-CAM for Multi-input Model
# ----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, inputs, class_idx=None, model_forward_fn=None):
        self.model.zero_grad()
        if model_forward_fn:
            output = model_forward_fn(inputs)
        else:
            output = self.model(*inputs) if isinstance(inputs, tuple) else self.model(inputs)

        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        one_hot = torch.zeros_like(logits)
        one_hot[0][class_idx] = 1
        logits.backward(gradient=one_hot, retain_graph=False)

        gradients = self.gradients
        activations = self.activations

        # Compute weights: global average pooling over spatial dims
        weight_dims = list(range(2, gradients.dim()))
        weights = torch.mean(gradients, dim=weight_dims)  # [B, C]

        # Expand weights to match activation shape
        expand_shape = [-1] + [1] * (activations.dim() - 2)
        weights = weights.view(expand_shape)

        cam = torch.sum(weights * activations, dim=1)  # [B, H, W] or [B, L]
        cam = torch.relu(cam).squeeze(0)  # remove batch
        return cam


def apply_colormap_on_image(signal_1d, cam_1d, colormap=cv2.COLORMAP_JET, height=250):
    """
    将 1D CAM 叠加到 ECG 信号上，生成可视化图像。
    - 线条更粗、更清晰
    - 画布更高，显示更舒展
    """
    # signal_1d = signal_1d[0:2450]
    # cam_1d = cam_1d[50:]
    length = len(signal_1d)
    
    # Normalize signal and cam
    signal_norm = (signal_1d - signal_1d.min()) / (signal_1d.max() - signal_1d.min() + 1e-8)
    cam_norm = (cam_1d - cam_1d.min()) / (cam_1d.max() - cam_1d.min() + 1e-8)

    # Create blank canvas
    img = np.ones((height, length, 3), dtype=np.uint8) * 255  # white background

    # Draw ECG waveform — thicker and more visible
    y_ecg = (1 - signal_norm) * (height - 30) + 15  # slightly more margin
    for i in range(1, length):
        cv2.line(
            img,
            (i - 1, int(y_ecg[i - 1])),
            (i, int(y_ecg[i])),
            color=(0, 0, 0),      # black, or try (0, 0, 200) for blue
            thickness=2,          # ← 关键：加粗线条！
            lineType=cv2.LINE_AA  # optional: anti-aliased for smoother look
        )

    # Expand 1D CAM to 2D
    cam_2d = np.tile(cam_norm, (height, 1))  # (height, length)

    # Apply colormap
    cam_uint8 = (cam_2d * 255).astype(np.uint8)
    cam_colored = cv2.applyColorMap(cam_uint8, colormap)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)

    # Overlay with heatmap
    alpha = 0.6
    overlay = cv2.addWeighted(img, 1 - alpha, cam_colored, alpha, 0)
    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/bgm/bgm_cpsc_afdb_1.yaml')
    parser.add_argument('--ckpt', type=str, default='checkpoint/bgm_cpsc_1.pth')
    parser.add_argument('--explain_n', type=int, default=100, help='Number of samples to explain')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # ----------------------------
    # Load Model
    # ----------------------------
    model = GMCNet()
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # ----------------------------
    # Setup Grad-CAM
    # ----------------------------
    time_target_layer = model.expand3_time[-2]  # Conv1d before ReLU
    freq_target_layer = model.expand3_freq[-2]  # Conv2d before ReLU

    grad_cam_time = GradCAM(model, time_target_layer)
    grad_cam_freq = GradCAM(model, freq_target_layer)

    result_dir = os.path.dirname(args.ckpt)
    explain_dir = os.path.join(result_dir, 'explanations')
    os.makedirs(explain_dir, exist_ok=True)
    print(f"Explanations will be saved to: {explain_dir}")

    # ----------------------------
    # Data Loader
    # ----------------------------
    data_cfg = config['dataset']['test']
    aug_cfg = config['augmentation']['test']
    split = data_cfg['split']

    transform = ECGAugment(**aug_cfg)

    dataset = AFDataset(
        path1=data_cfg['split_dir'],
        path2=data_cfg['data_dir'],
        split='test',
        fold=data_cfg['fold'],
        transform=transform,
        ga_test=True
    )

    test_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        drop_last=False
    )

    sample_count = 0

    def model_forward(inputs_pair):
        x_t, x_f = inputs_pair
        # with torch.no_grad():
        logits, _, _ = model(x_t, x_f)  
        return logits

    print("Starting inference and explanation...")
    # with torch.no_grad():
    model.train()
    for batch in test_loader:
        if len(batch) == 3:
            ecg, spec, labels = batch
        else:
            raise ValueError("Dataset must return (ecg, spectrogram, label)")

        ecg = ecg.to(device)
        spec = spec.to(device)
        labels = labels.to(device)

        # Forward pass
        logits, _, _ = model(ecg, spec)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)


        # Generate explanations
        if sample_count < args.explain_n:
            for i in range(ecg.size(0)):
                if sample_count >= args.explain_n:
                    break

                x_t = ecg[i:i+1]
                x_f = spec[i:i+1]
                y_true = labels[i].item()
                y_pred = preds[i].item()
                print(x_t.shape)
                print(x_f.shape)

                x_t.requires_grad_(True)
                x_f.requires_grad_(True)

                # Time-domain Grad-CAM
                try:
                    cam_t = grad_cam_time((x_t, x_f), class_idx=y_pred, model_forward_fn=model_forward)
                    cam_t = cam_t.cpu().numpy()
                    ecg_signal = x_t[0,:, 0].cpu().detach().numpy()  # [L]

                    if cam_t.shape[0] != ecg_signal.shape[0]:
                        # 使用线性插值上采样
                        cam_t = np.interp(
                            np.linspace(0, 1, ecg_signal.shape[0]),   # 目标位置
                            np.linspace(0, 1, cam_t.shape[0]),        # 原始位置
                            cam_t                                     # 原始值
                        )
                    print(cam_t.shape)
                    print(ecg_signal.shape)
                    overlay_t = apply_colormap_on_image(ecg_signal, cam_t, height=250)
                except Exception as e:
                    print(f"Failed to generate time CAM for sample {sample_count}: {e}")
                    overlay_t = np.ones((250, len(ecg_signal), 3), dtype=np.uint8) * 255

                # Frequency-domain Grad-CAM
                try:
                    cam_f = grad_cam_freq((x_t, x_f), class_idx=y_pred, model_forward_fn=model_forward)
                    cam_f = cam_f.cpu().numpy()
                    spec_img = x_f[0].cpu().detach().numpy()   # [C, H, W]
                    spec_gray = np.mean(spec_img, axis=0)  # [H, W]

                    H, W = spec_gray.shape
                    # 🔑 关键：将 CAM 上采样到原始频谱尺寸
                    if cam_f.shape != (H, W):
                        cam_f = cv2.resize(cam_f, (W, H), interpolation=cv2.INTER_LINEAR)

                    cam_f = (cam_f - cam_f.min()) / (cam_f.max() - cam_f.min() + 1e-8)
                    heatmap_f = cv2.applyColorMap((cam_f * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    heatmap_f = cv2.cvtColor(heatmap_f, cv2.COLOR_BGR2RGB)

                    spec_norm = ((spec_gray - spec_gray.min()) / (spec_gray.max() - spec_gray.min() + 1e-8) * 255).astype(np.uint8)
                    spec_rgb = cv2.cvtColor(spec_norm, cv2.COLOR_GRAY2BGR)  # [H, W, 3], BGR
                    spec_rgb = cv2.cvtColor(spec_rgb, cv2.COLOR_BGR2RGB)   # 转为 RGB，与 heatmap_f 一致

                    print(spec_rgb.shape)
                    print(heatmap_f.shape)
                    overlay_f = cv2.addWeighted(spec_rgb, 0.6, heatmap_f, 0.4, 0)
                except Exception as e:
                    print(f"Failed to generate freq CAM for sample {sample_count}: {e}")
                    overlay_f = np.ones((128, 128, 3), dtype=np.uint8) * 255

                # Save
                prefix = f'sample_{sample_count:04d}_true{y_true}_pred{y_pred}'
                cv2.imwrite(os.path.join(explain_dir, f'{prefix}_time.png'), cv2.cvtColor(overlay_t, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(explain_dir, f'{prefix}_freq.png'), overlay_f)

                sample_count += 1

    print(f"\nSaved {sample_count} explanation pairs to {explain_dir}")


if __name__ == '__main__':
    main()