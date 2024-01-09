import slideio
import cv2
from tqdm import tqdm

def extract_patches_from_svs(svs_path, patch_size, level=0, output_folder='C:/Users/minki.kim/Desktop/org/itp/result'):
    """
    SVS 파일에서 패치를 추출하는 함수입니다.

    :param svs_path: SVS 파일의 경로
    :param patch_size: 추출할 패치의 크기 (가로, 세로)
    :param level: 추출할 이미지의 레벨 (0은 가장 높은 해상도)
    :param output_folder: 추출한 패치를 저장할 폴더
    """
    # SVS 파일을 엽니다.
    slide = slideio.open_slide(svs_path, 'SVS')

    # 지정된 레벨의 슬라이드를 가져옵니다.
    scene = slide.get_scene(level)

    # 이미지의 전체 크기를 얻습니다.
    dims = scene.rect

    # 지정한 패치 크기로 이미지를 순회하며 패치를 추출합니다.
    for y in tqdm(range(0, dims[3], patch_size[1])):
        for x in range(0, dims[2], patch_size[0]):
            # 패치 이미지를 추출합니다.
            patch = scene.read_block((x, y, patch_size[0], patch_size[1]))
            
            # 추출한 패치를 파일로 저장합니다.
            patch_filename = f"{output_folder}/patch_{x}_{y}.png"
            cv2.imwrite(patch_filename, patch)

image_path = '320900-2023-06-00-02-01.SVS'

extract_patches_from_svs(image_path, (224, 224))
