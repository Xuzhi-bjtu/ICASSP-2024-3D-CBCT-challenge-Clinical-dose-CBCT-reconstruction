import torch
from torch import nn


class DoubleConv_Res(nn.Module):  # Conv3x3x3 -> BN -> LRelu -> Conv3x3x3 -> BN -> Residual sum -> LRelu
    def __init__(self, in_ch, mi_ch, out_ch):
        super(DoubleConv_Res, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, mi_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(mi_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(mi_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_ch)
        )
        self.ch_transform = None
        if in_ch != out_ch:
            self.ch_transform = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
                                              nn.BatchNorm3d(out_ch))
        self.LReLU = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = self.conv(x)
        if self.ch_transform is not None:
            x = self.ch_transform(x)
        out += x
        out = self.LReLU(out)
        return out


class DownConv(nn.Module):  # Conv3x3x3 (stride=2)
    def __init__(self, in_ch, out_ch):
        super(DownConv, self).__init__()
        self.downconv = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, input):
        return self.downconv(input)


class UpConv(nn.Module):  # ConvTrans3x3x3
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.upconv = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        )

    def forward(self, input):
        return self.upconv(input)


class Unet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(Unet3D, self).__init__()
        # Encoder
        self.conv1 = DoubleConv_Res(in_ch, 64, 64)
        self.down1 = DownConv(64, 64)
        self.conv2 = DoubleConv_Res(64, 128, 128)
        self.down2 = DownConv(128, 128)
        self.conv3 = DoubleConv_Res(128, 256, 256)
        self.down3 = DownConv(256, 256)
        self.conv4 = DoubleConv_Res(256, 512, 512)
        self.down4 = DownConv(512, 512)
        # Bottom
        self.conv5 = DoubleConv_Res(512, 1024, 512)
        # Decoder
        self.up1 = UpConv(512, 512)
        self.conv6 = DoubleConv_Res(1024, 512, 256)
        self.up2 = UpConv(256, 256)
        self.conv7 = DoubleConv_Res(512, 256, 128)
        self.up3 = UpConv(128, 128)
        self.conv8 = DoubleConv_Res(256, 128, 64)
        self.up4 = UpConv(64, 64)
        self.conv9 = DoubleConv_Res(128, 64, 64)
        # Output
        self.conv10 = nn.Conv3d(64, out_ch, kernel_size=3, stride=1, padding=1, bias=True)

        # Apply weight initialization to all modules
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv3d or type(m) == nn.ConvTranspose3d:
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')  # Kaiming (He)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm3d:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.Linear:
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1 = self.conv1(x)
        d1 = self.down1(e1)
        e2 = self.conv2(d1)
        d2 = self.down2(e2)
        e3 = self.conv3(d2)
        d3 = self.down3(e3)
        e4 = self.conv4(d3)
        d4 = self.down4(e4)
        b = self.conv5(d4)
        up1 = self.up1(b)
        merge1 = torch.cat([e4, up1], dim=1)
        de1 = self.conv6(merge1)
        up2 = self.up2(de1)
        merge2 = torch.cat([e3, up2], dim=1)
        de2 = self.conv7(merge2)
        up3 = self.up3(de2)
        merge3 = torch.cat([e2, up3], dim=1)
        de3 = self.conv8(merge3)
        up4 = self.up4(de3)
        merge4 = torch.cat([e1, up4], dim=1)
        de4 = self.conv9(merge4)
        out = self.conv10(de4)

        return out


if __name__ == '__main__':
    from torch import nn, optim
    import os

    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    model = Unet3D().to(device)
    model = nn.parallel.DataParallel(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    bs =1

    while 1:
        input = torch.randn((bs, 1, 128, 128, 128)).to(device)
        gt = torch.randn((bs, 1, 128, 128, 128)).to(device)

        # # ---------2023.7.25-----------
        # input = torch.randn((1, 1, 32, 512, 512)).to(device)
        # gt = torch.randn((1, 1, 32, 512, 512)).to(device)
        # # -----------------------------

        optimizer.zero_grad()
        # forward
        output = model(input)
        loss = criterion(output, gt)  # MSE loss
        # backward
        loss.backward()
        optimizer.step()
