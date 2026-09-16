import os
import cv2
import numpy as np

class OpticalFlowGenerator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_optical_flow(self, onset_frame, apex_frame):
        """
        Generates an optical flow image from the onset and apex frames.
        """
        if onset_frame is None or apex_frame is None:
            return None

        # Ensure frames are grayscale
        if len(onset_frame.shape) > 2:
            onset_frame = cv2.cvtColor(onset_frame, cv2.COLOR_BGR2GRAY)
        if len(apex_frame.shape) > 2:
            apex_frame = cv2.cvtColor(apex_frame, cv2.COLOR_BGR2GRAY)

        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(onset_frame, apex_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Convert flow to an HSV image
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((onset_frame.shape[0], onset_frame.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Convert HSV to BGR for saving
        bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr_flow

    def save_optical_flow_image(self, subject, sequence_name, flow_image):
        """
        Saves the optical flow image to the output directory.
        """
        output_path = os.path.join(self.output_dir, f"s{subject}_{sequence_name}_flow.jpg")
        cv2.imwrite(output_path, flow_image)
        return output_path