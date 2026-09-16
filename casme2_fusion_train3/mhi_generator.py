import cv2
import numpy as np
from PIL import Image
import os

class MHIGenerator:
    def __init__(self, duration=40, enable_face_alignment=True):
        """
        Motion History Image Generator with facial alignment
        Args:
            duration: Duration parameter for MHI (number of frames to consider)
            enable_face_alignment: Whether to enable face alignment using OpenCV
        """
        self.duration = duration
        self.enable_face_alignment = enable_face_alignment
        self.face_cascade = None  # Will be initialized when needed
    
    def detect_and_align_face(self, frame):
        """
        Detect face and align it for better MHI generation using OpenCV
        Args:
            frame: Input frame (grayscale)
        Returns:
            Aligned face region or original frame if no face detected
        """
        # Initialize face cascade if not already done (for multiprocessing compatibility)
        if self.face_cascade is None:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces using OpenCV Haar cascade
        faces = self.face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return frame  # Return original frame if no face detected
        
        # Use the largest face
        face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = face
        
        # Expand face region slightly to include more context
        margin = int(0.2 * min(w, h))
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(frame.shape[1] - x, w + 2 * margin)
        h = min(frame.shape[0] - y, h + 2 * margin)
        
        # Extract face region
        face_region = frame[y:y+h, x:x+w]
        
        # Resize face region to standard size for consistency
        if face_region.size > 0:
            face_region = cv2.resize(face_region, (224, 224))
        
        return face_region if face_region.size > 0 else frame
    
    def generate_mhi(self, frame_paths):
        """
        Generate Motion History Image from a sequence of frames
        Args:
            frame_paths: List of frame file paths in chronological order
        Returns:
            MHI as numpy array (single channel)
        """
        if len(frame_paths) < 2:
            raise ValueError("Need at least 2 frames to generate MHI")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(frame_paths[0], cv2.IMREAD_GRAYSCALE)
        if first_frame is None:
            raise ValueError(f"Could not read frame: {frame_paths[0]}")
        
        # Apply face detection and alignment to first frame
        if self.enable_face_alignment:
            first_frame_aligned = self.detect_and_align_face(first_frame)
        else:
            first_frame_aligned = first_frame
        
        height, width = first_frame_aligned.shape
        mhi = np.zeros((height, width), dtype=np.float32)
        
        prev_frame = first_frame_aligned.astype(np.float32)
        
        for i, frame_path in enumerate(frame_paths[1:], 1):
            current_frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if current_frame is None:
                continue
            
            # Apply face detection and alignment to current frame
            if self.enable_face_alignment:
                current_frame_aligned = self.detect_and_align_face(current_frame)
                if len(current_frame_aligned.shape) == 3:
                    current_frame_aligned = cv2.cvtColor(current_frame_aligned, cv2.COLOR_BGR2GRAY)
            else:
                current_frame_aligned = current_frame
            current_frame = current_frame_aligned.astype(np.float32)
            
            # Calculate frame difference
            diff = cv2.absdiff(current_frame, prev_frame)
            
            # Threshold the difference to get motion mask (lower threshold for micro-expressions)
            _, motion_mask = cv2.threshold(diff, 5, 1, cv2.THRESH_BINARY)
            
            # Update MHI
            # Decay existing values
            mhi = np.maximum(0, mhi - 1)
            
            # Set motion regions to current timestamp
            mhi[motion_mask == 1] = self.duration
            
            prev_frame = current_frame
        
        # Normalize MHI to 0-255 range
        if np.max(mhi) > 0:
            mhi = (mhi / np.max(mhi) * 255).astype(np.uint8)
        else:
            mhi = mhi.astype(np.uint8)
        
        return mhi
    
    def generate_mhi_from_sequence(self, sequence_path, onset, offset):
        """
        Generate MHI from a video sequence directory
        Args:
            sequence_path: Path to directory containing frame images
            onset: Start frame number
            offset: End frame number
        Returns:
            MHI as PIL Image (grayscale)
        """
        try:
            frame_files = sorted(os.listdir(sequence_path))
        except FileNotFoundError:
            raise FileNotFoundError(f"Directory not found: {sequence_path}")
        
        # Filter valid frames within onset-offset range
        valid_frames = []
        for f in frame_files:
            if f.startswith('img') and f.endswith('.jpg'):
                try:
                    frame_num = int(f[3:-4])
                    if onset <= frame_num <= offset:
                        valid_frames.append((frame_num, f))
                except ValueError:
                    continue
        
        if len(valid_frames) < 2:
            raise ValueError(f"Not enough valid frames found for {sequence_path} between {onset} and {offset}")
        
        # Sort by frame number and get file paths
        valid_frames.sort(key=lambda x: x[0])
        frame_paths = [os.path.join(sequence_path, f[1]) for f in valid_frames]
        
        # Generate MHI
        mhi_array = self.generate_mhi(frame_paths)
        
        # Convert to PIL Image
        mhi_image = Image.fromarray(mhi_array, mode='L')
        
        return mhi_image