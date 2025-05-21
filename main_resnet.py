import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import copy
import matplotlib.pyplot as plt

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach().clone()
        self.loss = 0.0

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach().clone()
        self.loss = 0.0

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

    @staticmethod
    def gram_matrix(input):
        b, c, h, w = input.size()
        features = input.view(b * c, h * w)
        G = torch.mm(features, features.t())
        return G.div(b * c * h * w)

class Normalization(nn.Module):
    def __init__(self, mean, std, device):
        super(Normalization, self).__init__()
        self.register_buffer('mean', mean.view(-1, 1, 1))
        self.register_buffer('std', std.view(-1, 1, 1))

    def forward(self, img):
        return (img - self.mean) / self.std

def load_image(img_path, device, imsize=512):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),
        transforms.ToTensor()])
    image = Image.open(img_path).convert('RGB')
    image = loader(image).unsqueeze(0).to(device, torch.float)
    return image

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=['layer_5'],
                               style_layers=['layer_1', 'layer_2', 'layer_3', 'layer_4']):
    device = style_img.device
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std, device).to(device)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        i += 1
        name = f'layer_{i}'
        if isinstance(layer, nn.ReLU):
            layer = nn.ReLU(inplace=False)
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f'content_loss_{i}', content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f'style_loss_{i}', style_loss)
            style_losses.append(style_loss)

    for idx in range(len(model) - 1, -1, -1):
        if isinstance(model[idx], ContentLoss) or isinstance(model[idx], StyleLoss):
            break
    model = model[:idx + 1]

    return model, style_losses, content_losses

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img,
                       num_steps=300, style_weight=1e6, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img)
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    style_loss_history = []
    content_loss_history = []
    run = [0]

    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_weight * style_score + content_weight * content_score
            loss.backward()
            style_loss_history.append(style_score.item() * style_weight)
            content_loss_history.append(content_score.item() * content_weight)
            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]}: Style Loss {style_score.item() * style_weight:.4f}, Content Loss {content_score.item():.4f}")
            return loss

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(style_loss_history, label='Style Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Style Loss Trend')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(content_loss_history, label='Content Loss', color='orange')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Content Loss Trend')
    plt.legend()

    plt.tight_layout()
    plt.savefig('loss_trend_resnet18.png')
    plt.show()

    return input_img

def load_backbone(name, device):
    if name == 'resnet18':
        resnet = models.resnet18(pretrained=True)
        cnn = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                            resnet.layer1, resnet.layer2, resnet.layer3).to(device).eval()
    else:
        raise ValueError(f"Unsupported model: {name}")

    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    return cnn, normalization_mean, normalization_std

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    content_path = 'content.jpeg'
    style_path = 'style.jpg'
    content_img = load_image(content_path, device)
    style_img = load_image(style_path, device)

    for model_name in ['resnet18']:
        print(f"\nRunning style transfer with {model_name}...")
        cnn, mean, std = load_backbone(model_name, device)
        input_img = content_img.clone().requires_grad_(True)

        start_time = time.time()
        output = run_style_transfer(cnn, mean, std, content_img, style_img, input_img)
        elapsed = time.time() - start_time

        unloader = transforms.ToPILImage()
        output_image = output.cpu().clone().squeeze(0)
        output_pil = unloader(output_image)
        output_path = f'output_{model_name}.png'
        output_pil.save(output_path)

        print(f'{model_name} style transfer complete: {output_path} saved. Time: {elapsed:.2f}s')
