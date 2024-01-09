import argparse
import torch
#from utils import set_seed
import cv2
from extractor.ctran import *
from torchsummary import summary


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = 'C:/Users/minki.kim/Desktop/org/itp/result/patch_35840_5376.png'
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    feature_extractor = ctranspath()
    feature_extractor.head = nn.Identity()
    td = torch.load('C:/Users/minki.kim/Desktop/org/extractor/ctranspath.pth')
    feature_extractor.load_state_dict(td['model'], strict=True)

    print(f"Before change extractor n(param) : {sum(p.numel() for p in model.parameters())}")
    model.feature_extractor = feature_extractor
    print(f"After changed extractor n(param) : {sum(p.numel() for p in model.parameters())}")


    for (k1, v1), (k2, v2) in zip(model.feature_extractor.state_dict().items(), feature_extractor.state_dict().items()):
        if torch.equal(v1, v2)==False:
            print(f"Layer {k1}: 파라미터가 다릅니다.")
            break
    

    results = model(img)

    detections = results.pred[0]
    print(len(detections))

    # 원본 이미지에 바운딩 박스를 그립니다.
    for *xyxy, conf, cls in detections:
        # 바운딩 박스의 좌표를 정수로 변환합니다.
        x1, y1, x2, y2 = map(int, xyxy)
        # 원본 이미지에 바운딩 박스를 그립니다.
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 빨간색 바운딩 박스

    save_path = 'C:/Users/minki.kim/Desktop/org/itp/detected_image.jpg'
    cv2.imwrite(save_path, img)
    print("done")

    '''
    results.render()  # Results의 바운딩 박스와 레이블을 이미지에 그립니다.
    
    # 결과 이미지를 저장할 경로를 지정합니다.
    save_path = 'C:/Users/minki.kim/Desktop/org/itp/result/detected_image.png'

    rendered_images = results.ims

    # 이미지를 파일로 저장합니다.
    for idx, img in enumerate(rendered_images):
        cv2.imwrite(f"{save_path[:-4]}_{idx}{save_path[-4:]}", img)
    '''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32,type=int)
    parser.add_argument("--seed", default=21, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--optimizer", default=1, type=int)
    
    main(parser)
   