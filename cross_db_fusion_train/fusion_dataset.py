import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN


class CrossDBFusionDataset(Dataset):
    """
    Dataset class for cross-database fusion training.
    Loads both RGB and Dynamic images from CASME II, SAMM, and SMIC.
    """
    
    def __init__(self, samples, rgb_transform=None, dynamic_transform=None, dynamic_image_dir=None):
        self.samples = samples
        self.rgb_transform = rgb_transform
        self.dynamic_transform = dynamic_transform
        self.dynamic_image_dir = dynamic_image_dir or './merged_dynamic_data'
        
        # Initialize MTCNN for face detection (for RGB images)
        self.mtcnn = MTCNN(keep_all=False, device='cpu')
    
    def __len__(self):
        return len(self.samples)
    
    def _extract_face_roi(self, image):
        """
        Extract face ROI using MTCNN.
        """
        try:
            boxes, _ = self.mtcnn.detect(image)
            if boxes is not None:
                box = boxes[0]
                x1, y1, x2, y2 = [int(b) for b in box]
                
                # Add some margin around the face
                margin = 20
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(image.width, x2 + margin)
                y2 = min(image.height, y2 + margin)
                
                # Crop the face region
                face_image = image.crop((x1, y1, x2, y2))
                return face_image
            else:
                # If no face detected, return the original image
                return image
        except Exception as e:
            print(f"Error in face detection: {e}")
            return image
    
    def _get_rgb_image(self, sample):
        sequence_path = sample['raw_video_path']
        dataset = sample['dataset']
        
        try:
            if dataset == 'smic':
                # SMIC uses .bmp files
                frame_files = sorted([f for f in os.listdir(sequence_path) if f.endswith('.bmp')])
                # SMIC doesn't have onset/offset in the same way, usually full sequence is micro-expression
                # But our parser might have handled it. Let's assume full sequence or specific logic.
                # In CombinedDataParser/SMICDataParser, we just have 'image_files' list?
                # Let's re-scan to be safe or use what's available.
                # For SMIC, we can just pick random frame from the sequence dir.
                if not frame_files:
                     return None
                selected_frame_file = random.choice(frame_files)
            
            elif dataset == 'samm':
                # SAMM uses .jpg files, name format: 006_1_2_001.jpg
                frame_files = sorted([f for f in os.listdir(sequence_path) if f.endswith('.jpg')])
                # Filter by onset/offset
                onset = sample['onset']
                offset = sample['offset']
                valid_frames = []
                for f in frame_files:
                     try:
                        # SAMM filenames end with _frame_number.jpg? 
                        # Actually SAMM filenames are like 006_1_2_001.jpg
                        # The last part is frame index.
                        frame_num = int(f.split('_')[-1].split('.')[0])
                        if onset <= frame_num <= offset:
                            valid_frames.append(f)
                     except ValueError:
                        continue
                if not valid_frames:
                     # Fallback to all frames if filtering fails
                     valid_frames = frame_files
                if not valid_frames: return None
                selected_frame_file = random.choice(valid_frames)

            elif dataset == 'casme2':
                # CASME2 uses imgXX.jpg
                frame_files = sorted([f for f in os.listdir(sequence_path) if f.endswith('.jpg') and f.startswith('img')])
                onset = sample['onset']
                offset = sample['offset']
                valid_frames = []
                for f in frame_files:
                    try:
                        frame_num = int(f[3:-4])
                        if onset <= frame_num <= offset:
                            valid_frames.append(f)
                    except ValueError:
                        continue
                if not valid_frames:
                     valid_frames = frame_files
                if not valid_frames: return None
                selected_frame_file = random.choice(valid_frames)
            
            else:
                return None

            rgb_image_path = os.path.join(sequence_path, selected_frame_file)
            rgb_image = Image.open(rgb_image_path).convert('RGB')
            return rgb_image
            
        except Exception as e:
            print(f"Error loading RGB for {dataset} sample {sequence_path}: {e}")
            return None

    def _get_dynamic_image(self, sample):
        dataset = sample['dataset']
        # Original subject ID (int or str)
        orig_sub = sample['original_subject'] 
        sequence_path = sample['raw_video_path']
        sequence_name = os.path.basename(sequence_path)
        
        dynamic_filename = None
        
        # Construct expected filename based on merge_data.py logic
        if dataset == 'casme2':
            # CASME2: dynamic_data/subXX/filename.jpg -> merged/casme2_filename.jpg
            # Based on file listing: casme2_s1_EP02_01f.jpg
            # Format is: casme2_s{int_id}_{sequence}.jpg
            sub_str = f"s{int(orig_sub)}"
            dynamic_filename = f"casme2_{sub_str}_{sequence_name}.jpg"
            
        elif dataset == 'samm':
            # SAMM: dynamic_data/subXX/filename.jpg -> merged/samm_filename.jpg
            # SAMM files usually: 006_1_2.jpg
            # Merged: samm_006_1_2.jpg
            dynamic_filename = f"samm_{sequence_name}.jpg"
            
        elif dataset == 'smic':
            # SMIC: smic_dynamic_data2/s1/filename.jpg -> merged/smic_filename.jpg
            # SMIC files usually: s1_micro_positive_s1_ep01.jpg
            # Wait, SMIC dynamic filenames are tricky.
            # Let's rely on finding the file that contains the sequence name.
            # Sequence name is unique enough? 
            # In SMIC parser: sequence = 's1_ep01' (example)
            # Dynamic file: s1_micro_positive_s1_ep01.jpg?
            # Or just s1_ep01.jpg?
            # Let's look at merge_data.py again. It copies everything.
            # We might need to search for the file in the directory.
            dynamic_filename = None
            # Search strategy
            search_pattern = f"smic_*{sequence_name}*.jpg" 
        
        # Try direct path first if constructed
        if dynamic_filename:
            dynamic_image_path = os.path.join(self.dynamic_image_dir, dynamic_filename)
            if os.path.exists(dynamic_image_path):
                try:
                    dynamic_image = Image.open(dynamic_image_path).convert('RGB')
                    return dynamic_image
                except Exception as e:
                    print(f"Error opening dynamic image {dynamic_image_path}: {e}")
                    return None
        
        # Fallback: Search in the directory
        # This is slower but safer given the naming variations
        # Search for a file that contains both dataset prefix and sequence name
        # For CASME2: "casme2" AND sequence_name (e.g. EP02_01f)
        # For SAMM: "samm" AND sequence_name (e.g. 006_1_2)
        # For SMIC: "smic" AND sequence_name (e.g. s1_ep01)
        
        candidates = []
        for f in os.listdir(self.dynamic_image_dir):
            if f.startswith(f"{dataset}_") and sequence_name in f:
                candidates.append(f)
        
        if not candidates:
            # print(f"No dynamic candidate found for {dataset} {sequence_name}")
            return None
        
        # Pick the best match (shortest one usually, or just the first)
        dynamic_image_path = os.path.join(self.dynamic_image_dir, candidates[0])
        
        try:
            dynamic_image = Image.open(dynamic_image_path).convert('RGB')
            return dynamic_image
        except Exception as e:
            print(f"Error opening dynamic image {dynamic_image_path}: {e}")
            return None

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample['label']
        
        # --- Load RGB Image ---
        rgb_image = self._get_rgb_image(sample)
        if rgb_image is None:
            # Return dummy if failed
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1
        
        # Extract face ROI (MTCNN)
        rgb_image = self._extract_face_roi(rgb_image)
        
        # --- Load Dynamic Image ---
        dynamic_image = self._get_dynamic_image(sample)
        if dynamic_image is None:
             # Return dummy
             return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1
        
        # Extract face ROI for dynamic image too
        dynamic_image = self._extract_face_roi(dynamic_image)
        
        # Apply transforms
        if self.rgb_transform:
            rgb_image = self.rgb_transform(rgb_image)
        
        if self.dynamic_transform:
            dynamic_image = self.dynamic_transform(dynamic_image)
        
        return rgb_image, dynamic_image, label
