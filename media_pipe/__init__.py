import os
import cv2
import numpy as np

from .mp_utils  import LMKExtractor
from .draw_util import FaceMeshVisualizer
from .pose_util import project_points_with_trans, matrix_to_euler_and_translation, euler_and_translation_to_matrix


class FaceMeshDetector:
    """
    Class for face mesh detection and landmark extraction.
    """
    def __init__(self) -> None:
        self.lmk_extractor = LMKExtractor()
        self.vis = FaceMeshVisualizer(forehead_edge=False, iris_edge=False, iris_point=True)
        
    def __call__(self, image: np.array):
        frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        try:
            face_result = self.lmk_extractor(frame_bgr)
            face_result['width'] = frame_bgr.shape[1]
            face_result['height'] = frame_bgr.shape[0]
        except:
            face_result = None

        if face_result is None:
            return np.zeros_like(frame_bgr), None

        lmks = face_result['lmks'].astype(np.float32)
        motion = self.vis.draw_landmarks((frame_bgr.shape[1], frame_bgr.shape[0]), lmks, normed=True)
        
        return motion, face_result


def smooth_pose_seq(pose_seq, window_size=5):
    smoothed_pose_seq = np.zeros_like(pose_seq)

    for i in range(len(pose_seq)):
        start = max(0, i - window_size // 2)
        end = min(len(pose_seq), i + window_size // 2 + 1)
        smoothed_pose_seq[i] = np.mean(pose_seq[start:end], axis=0)

    return smoothed_pose_seq
    

class FaceMeshAlign():

    def __init__(self):
        self.vis = FaceMeshVisualizer(forehead_edge=False, iris_edge=False, iris_point=True)

    def _scale_iris(self, lmks, verts):
        r_iris_ids = [468, 469, 470, 471, 472]
        l_iris_ids = [473, 474, 475, 476, 477]
        r_eye_ids = [33,7,163,144,145,153,154,155,246,161,160,159,158,157,173,133]
        l_eye_ids = [249,390,373,374,380,381,382,362,466,388,387,386,385,384,398,263]

        def scale_iris(iris_ids, eye_ids):
            iris_lmks, eye_lmks, eye_verts = lmks[iris_ids], lmks[eye_ids], verts[eye_ids]
            x0, y0, x1, y1 = np.min(eye_lmks[:, 0]), np.min(eye_lmks[:, 1]), np.max(eye_lmks[:, 0]), np.max(eye_lmks[:, 1])
            iris_lmks[:, 0] = (iris_lmks[:, 0] - x0) / (x1 - x0)
            iris_lmks[:, 1] = (iris_lmks[:, 1] - y0) / (y1 - y0)
            
            iris_verts = np.zeros((5, 2))
            x0, y0, x1, y1 = np.min(eye_verts[:, 0]), np.min(eye_verts[:, 1]), np.max(eye_verts[:, 0]), np.max(eye_verts[:, 1])
            iris_verts[:, 0] = iris_lmks[:, 0] * (x1 - x0) + x0
            iris_verts[:, 1] = iris_lmks[:, 1] * (y1 - y0) + y0

            return iris_verts

        r_iris_verts = scale_iris(r_iris_ids, r_eye_ids)
        l_iris_verts = scale_iris(l_iris_ids, l_eye_ids)
        verts = np.vstack((verts, r_iris_verts, l_iris_verts))
        return verts

    def __call__(self, ref_result, temp_results):
        width, height = ref_result['width'], ref_result['height']
        # prepare template data
        trans_mat_arr = np.array([x['trans_mat'] for x in temp_results])
        verts_arr = np.array([x['lmks3d'] for x in temp_results])
        bs_arr = np.array([x['bs'] for x in temp_results])
        min_bs_idx = np.argmin(bs_arr.sum(1))
        
        # compute delta pose
        pose_arr = np.zeros([trans_mat_arr.shape[0], 6])

        for i in range(pose_arr.shape[0]):
            euler_angles, translation_vector = matrix_to_euler_and_translation(trans_mat_arr[i]) # real pose of source
            pose_arr[i, :3] = euler_angles
            pose_arr[i, 3:6] = translation_vector
        
        init_tran_vec = ref_result['trans_mat'][:3, 3] # init translation of tgt
        pose_arr[:, 3:6] = pose_arr[:, 3:6] - pose_arr[0, 3:6] + init_tran_vec # (relative translation of source) + (init translation of tgt)

        pose_arr_smooth = smooth_pose_seq(pose_arr, window_size=1)
        pose_mat_smooth = [euler_and_translation_to_matrix(pose_arr_smooth[i][:3], pose_arr_smooth[i][3:6]) for i in range(pose_arr_smooth.shape[0])]    
        pose_mat_smooth = np.array(pose_mat_smooth)

        # face retarget
        verts_arr = verts_arr - verts_arr[min_bs_idx] + ref_result['lmks3d']
        # project 3D mesh to 2D landmark
        projected_vertices = project_points_with_trans(verts_arr, pose_mat_smooth, [height, width])
                
        pose_list = []
        for i, verts in enumerate(projected_vertices):
            verts = self._scale_iris(temp_results[i]['lmks'], verts)
            lmk_img = self.vis.draw_landmarks((width, height), verts, normed=False)
            pose_list.append(lmk_img)
        pose_list = np.array(pose_list)[1:]

        return pose_list