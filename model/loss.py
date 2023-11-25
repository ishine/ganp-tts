import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn
import torchvision

from torchvision.models.vgg import vgg16

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        #target = torch.unsqueeze(target, 1)
        #input = torch.unsqueeze(input, 1)
        #if input.shape[1] != 3:
        #    input = input.repeat(1, 3, 1, 1)
        #    target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(torch.exp(y_mag) - torch.exp(x_mag), p="fro") / torch.norm(torch.exp(y_mag), p="fro")


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_images, target_images):
        #target_images = torch.unsqueeze(target_images, 1).repeat(1, 3, 1, 1)
        #out_images = torch.unsqueeze(out_images, 1).repeat(1, 3, 1, 1)
        target_images = target_images.repeat(1, 3, 1, 1)
        out_images = out_images.repeat(1, 3, 1, 1)
        #target_images = target_images[:][:][:1000][:]
        #print("target_images",target_images.shape)   
        #print("out_images",out_images.shape)   
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        tv_loss = self.tv_loss(out_images)
        return  perception_loss, tv_loss
        #return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]



class FeatureLoss(nn.Module):
    def __init__(self, loss, blocks, weights, device):
        super().__init__()
        self.feature_loss = loss
        assert all(isinstance(w, (int, float)) for w in weights)
        assert len(weights) == len(blocks)

        self.weights = torch.tensor(weights).to(device)
        #VGG16 contains 5 blocks - 3 convolutions per block and 3 dense layers towards the end
        assert len(blocks) <= 5
        assert all(i in range(5) for i in blocks)
        assert sorted(blocks) == blocks

        vgg = vgg16_bn(pretrained=True).features
        vgg.eval()

        for param in vgg.parameters():
            param.requires_grad = False

        vgg = vgg.to(device)

        bns = [i - 2 for i, m in enumerate(vgg) if isinstance(m, nn.MaxPool2d)]
        assert all(isinstance(vgg[bn], nn.BatchNorm2d) for bn in bns)

        self.hooks = [FeatureHook(vgg[bns[i]]) for i in blocks]
        self.features = vgg[0: bns[blocks[-1]] + 1]

    def forward(self, inputs, targets):

        # normalize foreground pixels to ImageNet statistics for pre-trained VGG
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        inputs = F.normalize(inputs, mean, std)
        targets = F.normalize(targets, mean, std)

        # extract feature maps
        self.features(inputs)
        input_features = [hook.features.clone() for hook in self.hooks]

        self.features(targets)
        target_features = [hook.features for hook in self.hooks]

        loss = 0.0
        
        # compare their weighted loss
        for lhs, rhs, w in zip(input_features, target_features, self.weights):
            lhs = lhs.view(lhs.size(0), -1)
            rhs = rhs.view(rhs.size(0), -1)
            loss += self.feature_loss(lhs, rhs) * w

        return loss

class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.on)

    def on(self, module, inputs, outputs):
        self.features = outputs

    def close(self):
        self.hook.remove()
        
def perceptual_loss(x, y):
    F.mse_loss(x, y)
    
def PerceptualLoss(blocks, weights, device):
    return FeatureLoss(perceptual_loss, blocks, weights, device)

class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
     #   self.perceptual_loss = VGGPerceptualLoss()
        self.srgan_loss = GeneratorLoss()
        self.tv_loss = TVLoss()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.sc_loss = SpectralConvergengeLoss()

    def forward(self, inputs, predictions):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[6:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions
        mel_predictions_old = mel_predictions
        mel_targets_old = mel_targets
        postnet_mel_predictions_old = postnet_mel_predictions
        #print("mel_targets_old",mel_targets_old.shape)
        #print("mel_postnet_targets_predict_old",postnet_mel_predictions_old.shape)
        #print("mel_targets_predict_old",mel_predictions_old.shape)
        if mel_targets_old.shape[1] != mel_predictions_old.shape[1]:
               #print("mel_targets_old",mel_targets_old.shape)
               mel_targets_old1 = torch.split(mel_targets_old,1000,dim=1)
               mel_targets_old = mel_targets_old1[0]
               #postnet_mel_predictions_old = torch.split(mel_targets_old,1000,dim=1)
               #mel_targets_old = mel_targets_old[0]
              # print("mel_targets_old",mel_targets_old.shape)

        postnet_mel_images = torch.unsqueeze(postnet_mel_predictions_old, 1)
        mel_images = torch.unsqueeze(mel_predictions_old, 1) 
        target_mel_images = torch.unsqueeze(mel_targets_old, 1)
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))
        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        #postnet_mel_images = torch.log10(torch.unsqueeze(postnet_mel_predictions_old, 1)) * 20 
        #mel_images = torch.log10(torch.unsqueeze(mel_predictions_old, 1)) * 20 
        #target_mel_images = torch.log10(torch.unsqueeze(mel_targets_old, 1)) * 20 
        
        
        #perception_loss,tv_loss = self.srgan_loss(mel_images, target_mel_images)
        #perception_loss1,tv_loss1 = self.srgan_loss(postnet_mel_images, target_mel_images)
 #       perception_loss1 = self.perceptual_loss(postnet_mel_images, target_mel_images)
        #postnet_mel_images = torch.unsqueeze(postnet_mel_predictions, 1).repeat(1, 3, 1, 1)

  #      tv_loss1 = self.tv_loss(postnet_mel_images)
   #     tv_loss = self.tv_loss(mel_images)

        mel_loss_1 = self.sc_loss(mel_predictions, mel_targets) + self.sc_loss(postnet_mel_predictions, mel_targets)



        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

#        total_loss = (
 #           1.0 * (mel_loss + postnet_mel_loss) + duration_loss + pitch_loss + energy_loss +  0.006 * (perception_loss1+perception_loss)
  #      )
        total_loss = (
            1.0 * (mel_loss + postnet_mel_loss) + duration_loss + pitch_loss + energy_loss +  0.5 * mel_loss_1
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            mel_loss_1,
            mel_loss_1,
            pitch_loss,
            energy_loss,
            duration_loss,
        )
