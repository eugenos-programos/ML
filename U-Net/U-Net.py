import torchvision

def conv(in_channels, out_channels):
    seq_list = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels)
    ]
    return seq_list

def up_block(in_channels, out_channels, mode, last=False, n_classes=None):
    seq_list = []
    if mode == "first":
        up_elem = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        ) 
    if mode == "second":
        up_elem = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
    seq_list += conv(in_channels, out_channels)
    if last:
        seq_list.append(nn.Conv2d(64, 1, 1, 1))
    return nn.Sequential(*seq_list), up_elem

def down_block(in_channels, out_channels, mode):
    seq_list = conv(in_channels, out_channels)
    if mode == "first":
        pooling = nn.MaxPool2d(2)
    else:
        pooling = nn.Conv2d(out_channels, out_channels, 2, 2)
    return nn.Sequential(*seq_list), pooling

def crop(enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs    

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, mode="first"):
        super(UNet, self).__init__()
        
        self.down1, self.pool1 = down_block(in_channels, 64, mode)
        self.down2, self.pool2 = down_block(64, 128, mode)
        self.down3, self.pool3 = down_block(128, 256, mode)
        self.down4, self.pool4 = down_block(256, 512, mode)
        self.bottle = nn.Sequential(*conv(512, 1024))
        self.up4, self.up_elem4 = up_block(1024, 512, mode)
        self.up3, self.up_elem3 = up_block(512, 256, mode)
        self.up2, self.up_elem2 = up_block(256, 128, mode)
        self.up1, self.up_elem1 = up_block(128, 64, mode, True, out_channels)
        
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        # bottleneck
        d_bottle = self.up_elem4(self.bottle(self.pool4(d4)))
        # up block - 1
        d4 = crop(d4, d_bottle)
        up5 = self.up4(torch.cat([d4, d_bottle], dim=1))
        # up block - 2
        up4 = self.up_elem3(up5)
        d3 = crop(d3, up4)
        up4 = self.up3(torch.cat([d3, up4], dim=1))
        # up block - 3
        up3 = self.up_elem2(up4)
        d2 = crop(d2, up3)
        up3 = self.up2(torch.cat([d2, up3], dim=1))
        # up block - 4
        up2 = self.up_elem1(up3)
        d1 = crop(d1, up2)
        up2 = self.up1(torch.cat([d1, up2], dim=1))
        return up2
