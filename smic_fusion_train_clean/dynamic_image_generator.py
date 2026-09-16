import os
import cv2
import numpy as np

class DynamicImageGenerator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_dynamic_image(self, image_sequence):
        """
        Generates a dynamic image from a sequence of images.
        A dynamic image is a weighted sum of the frames in a sequence.
        """
        num_frames = len(image_sequence)
        if num_frames == 0:
            return None

        # Calculate weights (coefficients) for each frame
        coefficients = np.zeros(num_frames)
        for i in range(num_frames):
            coefficients[i] = np.sum([(2 * (j + 1) - num_frames - 1) / (j + 1) for j in range(i, num_frames)])

        # Normalize coefficients
        coefficients /= np.sum(np.abs(coefficients))

        # Create the dynamic image
        first_frame = image_sequence[0]
        dynamic_image = np.zeros_like(first_frame, dtype=np.float32)

        for i, frame in enumerate(image_sequence):
            dynamic_image += coefficients[i] * frame

        # Normalize the dynamic image to 0-255 and convert to uint8
        dynamic_image = cv2.normalize(dynamic_image, None, 0, 255, cv2.NORM_MINMAX)
        dynamic_image = np.uint8(dynamic_image)

        return dynamic_image

    def save_dynamic_image(self, subject, sequence_name, dynamic_image):
        """
        Saves the dynamic image to the output directory organized by subject.
        """
        # Create subject directory if it doesn't exist
        subject_dir = os.path.join(self.output_dir, f"s{subject}")
        if not os.path.exists(subject_dir):
            os.makedirs(subject_dir)
        
        output_path = os.path.join(subject_dir, f"s{subject}_{sequence_name}.jpg")
        cv2.imwrite(output_path, dynamic_image)
        return output_path