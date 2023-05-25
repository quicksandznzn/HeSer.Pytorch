import os

import cv2

from face_tools.LVT import Engine
from face_tools.utils import utils




def face_mask(engine, file_path: str):
    img = cv2.imread(file_path)
    # detect face
    p_img = engine.preprocess_face(img, True)
    loc, conf, iou = engine.get_face(p_img)
    # only select first box
    bboxes, kpss = engine.postprocess_face(img, loc, conf, iou)
    # lmk
    crop_img, top, crop_height = utils.crop_image(img, bboxes[0])
    # get parsing
    p_img = engine.preprocess_parsing(crop_img)
    parsing = engine.get_parsing(p_img)
    parsing = engine.postprocess_parsing(parsing, *crop_img.shape[:2])
    mask_path = file_path.replace("/dataset/process/img/", "/dataset/process/mask/")
    mask_dir = os.path.dirname(mask_path)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    print(mask_path)
    cv2.imwrite(mask_path, parsing)


if __name__ == '__main__':
    base = "../dataset/process/img"
    img = cv2.imread('../face_tools/images/barack-obama-gty-jt-210802_1627927668233_hpMain_16x9_1600.jpeg')
    engine = Engine(face_detector_path='../face_tools/weights/yunet_final_dynamic_simplify.onnx'
                    , face_lmk_path='../face_tools/weights/slpt-lmk.onnx'
                    , gender_path='../face_tools/weights/fairface.onnx'
                    , deep3d_path='../face_tools/weights/deep3d.onnx'
                    , bfm_path='../face_tools/weights/postdeep3d.onnx'
                    , face_id_path='../face_tools/weights/id_model.onnx'
                    , face_parsing_path='../face_tools/weights/face_parsing.pt')

    for idname in os.listdir(base):
        id_path = os.path.join(base, idname)
        for video_clip in os.listdir(id_path):
            path = os.path.join(id_path, video_clip)
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                if '.png' in file_path:
                    face_mask(engine,file_path)
