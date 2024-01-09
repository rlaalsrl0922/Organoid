import argparse
import torch
from utils import set_seed
from yolov5 import YOLOv5

def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLOv5.load('yolov5s')
     # 이미지를 로드하고 전처리합니다.
    img = 'C:/Users/minki.kim/Desktop/org/itp/result/patch_35840_5376.png'

    # 이미지에서 객체 탐지를 수행합니다.
    results = model(img)

    # 결과를 출력합니다.
    results.print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32,type=int)
    parser.add_argument("--seed", default=21, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--optimizer", default=1, type=int)
    
    main(parser)
   