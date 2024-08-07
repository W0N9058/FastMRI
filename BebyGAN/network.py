import torch
import torch.nn as nn
from utils.model.varnet import VarNet
from utils.modules.discriminator import Discriminator_VGG_192
from utils.loss import AdversarialLoss, PerceptualLoss, BBL

class Generator(VarNet):
    def __init__(self):
        super(Generator, self).__init__()

class Discriminator(Discriminator_VGG_192):
    def __init__(self, in_chl, nf):
        super(Discriminator, self).__init__(in_chl=in_chl, nf=nf)

class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        self.G = Generator()
        self.D = Discriminator(in_chl=config.MODEL.D.IN_CHANNEL, nf=config.MODEL.D.N_CHANNEL)

        self.recon_loss_weight = config.MODEL.BBL_WEIGHT
        self.adv_loss_weight = config.MODEL.get('ADV_LOSS_WEIGHT', 0.1)
        self.bp_loss_weight = config.MODEL.get('BACK_PROJECTION_LOSS_WEIGHT', 0.1)
        self.use_pcp = config.MODEL.get('USE_PCP_LOSS', False)

        self.recon_criterion = BBL(
            alpha=config.MODEL.get('BBL_ALPHA', 1.0),
            beta=config.MODEL.get('BBL_BETA', 1.0),
            ksize=config.MODEL.get('BBL_KSIZE', 3),
            pad=config.MODEL.get('BBL_PAD', 1),
            stride=config.MODEL.get('BBL_STRIDE', 1),
            criterion=config.MODEL.get('BBL_TYPE', 'l1')
        )
        self.adv_criterion = AdversarialLoss(gan_type=config.MODEL.D.get('LOSS_TYPE', 'lsgan'))
        self.bp_criterion = nn.L1Loss(reduction='mean')

        if self.use_pcp:
            self.pcp_criterion = PerceptualLoss(
                layer_weights=config.MODEL.get('VGG_LAYER_WEIGHTS', None),
                vgg_type=config.MODEL.get('VGG_TYPE', 'vgg19'),
                use_input_norm=config.MODEL.get('USE_INPUT_NORM', True),
                use_pcp_loss=config.MODEL.get('USE_PCP_LOSS', False),
                use_style_loss=config.MODEL.get('USE_STYLE_LOSS', False),
                norm_img=config.MODEL.get('NORM_IMG', True),
                criterion=config.MODEL.get('PCP_LOSS_TYPE', 'l1')
            )
            self.pcp_loss_weight = config.MODEL.get('PCP_LOSS_WEIGHT', 0.1)
            self.style_loss_weight = config.MODEL.get('STYLE_LOSS_WEIGHT', 0.1)

    def forward(self, x, mask=None):  # mask를 기본값으로 None으로 설정
        # Generator를 사용하여 이미지를 생성
        generated = self.G(x, mask)
        
        # 생성된 이미지를 Discriminator에 전달하여 판별
        disc_output = self.D(generated)

        return generated, disc_output

if __name__ == '__main__':
    from config import config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Network(config)
    net.to(device)
    print("model have {:.3f}M parameters in total".format(sum(x.numel() for x in net.G.parameters()) / 1e6))