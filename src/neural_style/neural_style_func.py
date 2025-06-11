import argparse
import os
import sys
import time
import re
import cv2
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.onnx

from .utils import *
from .transformer_net import *
from .vgg import *


def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not os.path.exists(args.checkpoint_model_dir):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args):
    # 使用 torch.cuda 判断 GPU 可用性
    if args.accel and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    transformer = TransformerNet().to(device)
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = load_image(args.style_image, size=args.style_size)
    style = style_transform(style).repeat(args.batch_size, 1, 1, 1).to(device)

    features_style = vgg(normalize_batch(style))
    gram_style = [gram_matrix(y) for y in features_style]

    for e in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            y = normalize_batch(y)
            x = normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = args.content_weight * mse_loss(
                features_y.relu2_2, features_x.relu2_2
            )

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = (
                    f"{time.ctime()}\tEpoch {e + 1}:\t[{count}/{len(train_dataset)}]"
                    f"\tcontent: {agg_content_loss / (batch_id + 1):.6f}"
                    f"\tstyle: {agg_style_loss / (batch_id + 1):.6f}"
                    f"\ttotal: {(agg_content_loss + agg_style_loss) / (batch_id + 1):.6f}"
                )
                print(mesg)

            if args.checkpoint_model_dir and (batch_id + 1) % args.checkpoint_interval == 0:
                transformer.eval().cpu()
                ckpt_name = f"ckpt_epoch_{e}_batch_{batch_id + 1}.pth"
                ckpt_path = os.path.join(args.checkpoint_model_dir, ckpt_name)
                torch.save(transformer.state_dict(), ckpt_path)
                transformer.to(device).train()

    # 保存最终模型
    transformer.eval().cpu()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_name = f"epoch{args.epochs}_{timestamp}_cw{args.content_weight}_sw{args.style_weight}.pth"
    save_path = os.path.join(args.save_model_dir, save_name)
    torch.save(transformer.state_dict(), save_path)
    print("Done, trained model saved at", save_path)


def stylize(args, content_image_cv2):
    # 使用 torch.cuda 判断 GPU
    if args.accel and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # 将 cv2.imread 生成的对象转换为适合模型输入的格式
    content_image = cv2.cvtColor(content_image_cv2, cv2.COLOR_BGR2RGB)
    content_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)

    if args.model.endswith(".onnx"):
        output = stylize_onnx(content_image, args)
    else:
        with torch.no_grad():
            style_model = TransformerNet()
            state_dict = torch.load(args.model, map_location=device)
            # 移除过期的 InstanceNorm 运行统计
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device).eval()
            if args.export_onnx:
                assert args.export_onnx.endswith(".onnx"), "Export file must end with .onnx"
                output = torch.onnx._export(
                    style_model, content_image, args.export_onnx, opset_version=11
                ).cpu()
            else:
                output = style_model(content_image).cpu()

    # 将输出转换为适合 cv2.imshow 的格式
    output_image = output[0].permute(1, 2, 0).numpy()
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    return output_image


def stylize_onnx(content_image, args):
    import onnxruntime

    ort_session = onnxruntime.InferenceSession(args.model)

    def to_numpy(tensor):
        return tensor.cpu().numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(content_image)}
    ort_outs = ort_session.run(None, ort_inputs)
    return torch.from_numpy(ort_outs[0])


# 新增入口函数
def stylize_image(content_image_cv2, model_path, accel=True):
    class Args:
        def __init__(self):
            self.model = model_path
            self.content_scale = None
            self.export_onnx = None
            self.accel = accel

    args = Args()
    return stylize(args, content_image_cv2)


def main():
    parser = argparse.ArgumentParser(description="parser for fast - neural - style")
    subs = parser.add_subparsers(dest="subcommand", title="subcommands")

    train_p = subs.add_parser("train", help="parser for training arguments")
    train_p.add_argument("--epochs", type=int, default=2, help="number of training epochs")
    train_p.add_argument("--batch-size", type=int, default=4, help="batch size for training")
    train_p.add_argument("--dataset", type=str, required=True, help="path to training dataset")
    train_p.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg", help="path to style-image")
    train_p.add_argument("--save-model-dir", type=str, required=True, help="folder to save trained model")
    train_p.add_argument("--checkpoint-model-dir", type=str, default=None, help="folder to save checkpoints")
    train_p.add_argument("--image-size", type=int, default=256, help="size of training images")
    train_p.add_argument("--style-size", type=int, default=None, help="size of style-image")
    train_p.add_argument("--content-weight", type=float, default=1e5, help="weight for content-loss")
    train_p.add_argument("--style-weight", type=float, default=1e10, help="weight for style-loss")
    train_p.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    train_p.add_argument("--log-interval", type=int, default=500, help="log interval")
    train_p.add_argument("--checkpoint-interval", type=int, default=2000, help="checkpoint interval")
    train_p.add_argument("--seed", type=int, default=42, help="random seed")
    train_p.add_argument("--accel", action="store_true", help="use GPU acceleration")

    eval_p = subs.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_p.add_argument("--content-image", type=str, required=True, help="path to content image")
    eval_p.add_argument("--content-scale", type=float, default=None, help="factor to scale down content image")
    eval_p.add_argument("--model", type=str, required=True, help="model file to use (.pth or .onnx)")
    eval_p.add_argument("--export_onnx", type=str, help="export ONNX model path")
    eval_p.add_argument("--accel", action="store_true", help="use GPU acceleration")

    args = parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        content_image_cv2 = cv2.imread(args.content_image)
        output_image = stylize_image(content_image_cv2, args.model, args.accel)
        cv2.imshow('Stylized Image', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
