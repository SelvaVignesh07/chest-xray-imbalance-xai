"""
gradcam.py
==========
Phase 5: Explainable AI — Grad-CAM (Gradient-weighted Class Activation Mapping)

Implemented from scratch using PyTorch forward/backward hooks, following
Selvaraju et al. (2017), "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization."

How Grad-CAM works, conceptually:
    1. Run a forward pass, capturing the activation maps A^k of a chosen
       convolutional layer (for ResNet, the last conv block `layer4` is
       standard -- it has the richest high-level semantic features while
       still retaining spatial resolution to localize).
    2. Run a backward pass from the score for the TARGET class (e.g.
       PNEUMONIA), capturing the gradients dY^c/dA^k flowing into that same
       layer.
    3. Global-average-pool the gradients over spatial dimensions to get an
       importance weight alpha_k per channel: "how much does channel k
       matter for THIS class's prediction, on average?"
    4. Compute a weighted combination of the activation maps using those
       alpha_k weights, then apply ReLU (we only care about features that
       have a POSITIVE influence on the target class -- this is what
       "gradient-weighted" + "class activation" means together).
    5. Upsample this coarse heatmap to the input image resolution and
       overlay it, producing a visual explanation of which pixels the
       network relied on most for its decision.

This directly relates to the paper you're reproducing: Grad-CAM is one of
the core XAI techniques evaluated there through a human-centered lens (i.e.
"does this heatmap actually highlight clinically relevant lung regions, as
judged by radiologists?").
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Reusable Grad-CAM wrapper around a trained classification model.

    Usage:
        cam = GradCAM(model, target_layer=model.layer4[-1])
        heatmap = cam.generate(input_tensor, target_class=1)
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model (nn.Module): trained model in eval mode.
            target_layer (nn.Module): the convolutional layer to hook into.
                For torchvision ResNet, `model.layer4[-1]` (the last
                bottleneck/basic block of the last residual stage) is the
                conventional choice -- deep enough for semantic meaning,
                still spatial enough (7x7 for a 224x224 input) for
                localization.
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Forward hook: stashes the layer's output activations every time
        # this layer is executed during a forward pass.
        self.target_layer.register_forward_hook(self._save_activations)
        # Full backward hook: stashes the gradient flowing INTO this layer's
        # output during backpropagation (i.e. dLoss/dActivation, but here we
        # backprop from the class score rather than a loss).
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        # grad_output is a tuple; index 0 corresponds to the gradient w.r.t.
        # this layer's output tensor.
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Compute the Grad-CAM heatmap for a single input image.

        Args:
            input_tensor (torch.Tensor): shape [1, 3, 224, 224], already
                normalized and on the correct device.
            target_class (int or None): class index to explain. If None,
                uses the model's own top predicted class (useful for
                "explain whatever the model actually decided").

        Returns:
            np.ndarray: heatmap of shape [224, 224], values normalized to
                [0, 1], where higher = more influential for the target class.
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        # --- Forward pass ---
        output = self.model(input_tensor)  # shape [1, num_classes]

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # --- Backward pass from the target class's raw score ---
        # We zero all other gradients and backprop only the scalar score for
        # the class we want to explain -- this is what makes the resulting
        # heatmap CLASS-SPECIFIC (e.g. "what made the model think
        # PNEUMONIA" rather than a generic saliency map).
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        # activations: [1, C, H, W]   gradients: [1, C, H, W]
        activations = self.activations[0]   # [C, H, W]
        gradients = self.gradients[0]        # [C, H, W]

        # Global-average-pool gradients over spatial dims -> per-channel
        # importance weight alpha_k, shape [C].
        alpha = gradients.mean(dim=(1, 2))

        # Weighted sum of activation maps along the channel dimension.
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32,
                           device=activations.device)  # [H, W]
        for k, w in enumerate(alpha):
            cam += w * activations[k]

        # ReLU: keep only features with a POSITIVE contribution to the
        # target class's score -- negative-influence regions are not part
        # of "why the model predicted this class."
        cam = F.relu(cam)

        # Normalize to [0, 1] for visualization.
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cam.cpu().numpy()

        # Upsample from the conv layer's small spatial size (e.g. 7x7) to
        # the full 224x224 input resolution.
        cam_resized = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        return cam_resized


def overlay_heatmap_on_image(original_image_pil, heatmap, alpha=0.45):
    """
    Blend a Grad-CAM heatmap on top of the original (un-normalized) image.

    Args:
        original_image_pil (PIL.Image): the RAW image (not normalized),
            resized to match the heatmap's resolution (224x224).
        heatmap (np.ndarray): [H, W] values in [0, 1] from GradCAM.generate.
        alpha (float): blend strength of the heatmap overlay.

    Returns:
        np.ndarray: RGB uint8 image (H, W, 3) with heatmap overlaid, ready
            to save with cv2.imwrite or plt.imsave.
    """
    original_np = np.array(original_image_pil.convert("RGB"))

    heatmap_uint8 = np.uint8(255 * heatmap)
    # Apply a color map (JET: blue = low importance, red = high importance --
    # the standard convention in the Grad-CAM paper and most XAI literature).
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlaid = np.uint8(original_np * (1 - alpha) + heatmap_color * alpha)
    return overlaid


def generate_and_save_gradcam(model, target_layer, image_path, eval_transform,
                               device, target_class, class_names, save_path):
    """
    End-to-end convenience function: load an image from disk, run Grad-CAM,
    overlay the heatmap, and save the result as a PNG.

    Args:
        model (nn.Module): trained model.
        target_layer (nn.Module): e.g. model.layer4[-1].
        image_path (str): path to a sample chest X-ray image (e.g. a True
            Positive PNEUMONIA case, chosen deliberately so the heatmap can
            be visually checked against clinically expected lung regions).
        eval_transform (torchvision.transforms.Compose): the SAME
            normalization used at test time (from data_prep.get_transforms).
        device (torch.device): CUDA / MPS / CPU.
        target_class (int): class index to explain (typically the
            PNEUMONIA index).
        class_names (list[str]): for labeling/printing.
        save_path (str): output PNG path.
    """
    original_image = Image.open(image_path).convert("RGB")
    # Resize a copy of the RAW (un-normalized) image for visualization --
    # this is what gets shown to the user, NOT the normalized tensor.
    display_image = original_image.resize((224, 224))

    input_tensor = eval_transform(original_image).unsqueeze(0).to(device)

    cam = GradCAM(model, target_layer)
    heatmap = cam.generate(input_tensor, target_class=target_class)
    overlaid = overlay_heatmap_on_image(display_image, heatmap)

    Image.fromarray(overlaid).save(save_path)
    print(f"Saved Grad-CAM overlay for class '{class_names[target_class]}' to {save_path}")
    return overlaid
