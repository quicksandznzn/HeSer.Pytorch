'''
@author: LeslieZhao
@Date: 20220712
'''
from face_tools.model.face_detector import FaceDetector
from face_tools.model.face_id import FaceID
from face_tools.model.deep3d import Deep3d
from face_tools.model.face_lmk import FaceLmk
from face_tools.model.face_gender import FaceGender
from face_tools.model.face_parsing import FaceParsing


class Engine(FaceDetector,
            FaceParsing,
            FaceGender,
            FaceID,
            Deep3d,
            FaceLmk):
    def __init__(self,face_detector_path=None,
                     face_lmk_path=None,
                      gender_path=None,
                      deep3d_path=None,
                      bfm_path=None,
                      face_id_path=None,
                      face_parsing_path=None):

        
        if face_detector_path is not None:
            FaceDetector.__init__(self,face_detector_path)
            
        if face_lmk_path is not None:
            FaceLmk.__init__(self,face_lmk_path)
        #     self.face_lmk_detector = self.face_lmk_detector = ort.InferenceSession(
        #                     onnx.load(face_lmk_path).SerializeToString(),
        #                     providers=[
        #                     'CUDAExecutionProvider',
        #                     'CPUExecutionProvider'])

        if gender_path is not None:
        
            FaceGender.__init__(self,gender_path)

        if deep3d_path is not None:
            Deep3d.__init__(self,deep3d_path,bfm_path)

        if face_id_path is not None:
            FaceID.__init__(self,face_id_path)

        if face_parsing_path is not None:
            
            FaceParsing.__init__(self,face_parsing_path)

   