import torch
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

# ─── 1. GradCAM++ ────────────────────────────────────────────────────────────
class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        grad = self.gradients          # [1, C, H, W]
        act  = self.activations        # [1, C, H, W]

        # ── GradCAM++ weight calculation ──────────────────────────────────
        grad_sq   = grad ** 2
        grad_cu   = grad ** 3
        # Denominator: sum over spatial dimensions of 2*grad² + act * grad³
        denom = 2 * grad_sq + act * grad_cu
        denom = torch.where(denom != 0, denom,
                            torch.ones_like(denom))   # Avoid division by zero

        alpha  = grad_sq / denom                      # [1, C, H, W]
        # ReLU(grad) is used to only consider positive gradient directions
        weight = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)  # [1,C,1,1]

        cam = (weight * act).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam).squeeze().cpu().numpy()

        # ── Normalization ─────────────────────────────────────────────────
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_class


# ─── 2. Heatmap post-processing: focus ──────────────────────────────────────
def focus_heatmap(cam, blur_ksize=11, threshold=0.4, power=2.0):
    """
    blur_ksize : Gaussian kernel size (larger = smoother)
    threshold  : Values below this are zeroed out to highlight high‑activation areas
    power      : Power exponent to enhance contrast (>1 makes high values more prominent)
    """
    # Gaussian smoothing
    cam_blur = cv2.GaussianBlur(cam, (blur_ksize, blur_ksize), 0)
    # Thresholding
    cam_blur[cam_blur < threshold] = 0
    # Power contrast enhancement
    cam_blur = cam_blur ** power
    # Re-normalize
    if cam_blur.max() > 0:
        cam_blur = (cam_blur - cam_blur.min()) / (cam_blur.max() - cam_blur.min() + 1e-8)
    return cam_blur


# ─── 3. Load model ──────────────────────────────────────────────────────────
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()

target_layer = model.layer4[1].conv2
gradcampp = GradCAMPlusPlus(model, target_layer)

# ─── 4. Load and preprocess image ──────────────────────────────────────────
img_path = 'E:/RTi-YOLO/data/images/img000713.jpg'
img_pil  = Image.open(img_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
img_tensor = transform(img_pil).unsqueeze(0)

# ─── 5. Generate and focus heatmap ─────────────────────────────────────────
cam_raw, pred_class = gradcampp.generate(img_tensor)
cam_focused = focus_heatmap(
    cv2.resize(cam_raw, (224, 224)),
    blur_ksize=11,    # Tunable: smaller keeps more detail
    threshold=0.4,    # Tunable: higher gives more focus (suggested 0.3~0.6)
    power=2.0         # Tunable: larger emphasises peak regions more
)

# ─── 6. Pseudo‑color and overlay ──────────────────────────────────────────
heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_focused), cv2.COLORMAP_JET)
heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

img_crop = img_pil.resize((256, 256)).crop((16, 16, 240, 240))
img_np   = np.array(img_crop)
overlay  = cv2.addWeighted(img_np, 0.5, heatmap_rgb, 0.5, 0)

# ─── 7. Visualization ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_np)
axes[0].set_title('Original Image', fontsize=13)
axes[0].axis('off')

axes[1].imshow(heatmap_rgb)
axes[1].set_title(f'GradCAM++ (class={pred_class})', fontsize=13)
axes[1].axis('off')

axes[2].imshow(overlay)
axes[2].set_title('Focused Overlay', fontsize=13)
axes[2].axis('off')

plt.tight_layout()
plt.savefig('gradcampp_focused.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Predicted class index: {pred_class}")