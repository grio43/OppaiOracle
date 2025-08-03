#!/usr/bin/env python3
"""
Anime Image Tagger - Teacher Feature Extraction Pipeline
Preprocesses images and extracts features from dual teachers for training
Correctly handles different preprocessing requirements for each model
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import h5py
import webp
from tqdm import tqdm
import transformers
from safetensors import safe_open

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing pipeline"""
    # Storage locations in priority order
    storage_locations = [
        {"path": "/home/user/datasets/anime_curated", "priority": 0, "type": "local"},
        {"path": "/mnt/das/anime_archive", "priority": 1, "type": "das"},
        {"path": "/mnt/nas/anime_dataset/primary", "priority": 2, "type": "nas"},
        {"path": "/mnt/nas/anime_dataset/video_frames", "priority": 3, "type": "nas"}
    ]
    
    # Model paths
    anime_teacher_path: str = "/models/anime_tag_model_70k"
    clip_model_id: str = "zer0int/CLIP-SAE-ViT-L-14"
    
    # Model-specific image sizes
    student_image_size: int = 640  # Our model
    anime_teacher_size: int = 512  # Anime model trained on 512
    # CLIP size handled by processor (224)
    
    # Processing settings
    anime_batch_size: int = 256
    clip_batch_size: int = 128  
    num_workers: int = 8
    
    # Output settings
    output_dir: str = "/home/user/datasets/teacher_features"
    chunk_size: int = 10000
    
    # GPU allocation
    anime_teacher_gpu: int = 0
    clip_teacher_gpu: int = 1
    
    # Gray padding color
    pad_color: Tuple[int, int, int] = (114, 114, 114)


class ImagePreprocessor:
    """Handles image loading and model-specific preprocessing"""
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
        
        # For student model (640x640 letterboxed)
        self.student_size = config.student_image_size
        self.pad_color = config.pad_color
        
        # For anime teacher (512x512)
        self.anime_teacher_size = config.anime_teacher_size
        
        # Normalization for anime teacher (typical for anime models)
        self.anime_normalize = T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        
        # CLIP processor handles its own preprocessing
        from transformers import CLIPProcessor
        self.clip_processor = CLIPProcessor.from_pretrained(
            config.clip_model_id, torch_dtype=torch.float16
        )
    
    def load_image(self, path: Union[str, Path]) -> Image.Image:
        """Load image handling WebP and other formats"""
        path = Path(path)
        
        if path.suffix.lower() == '.webp':
            # Use webp library for better handling
            try:
                with open(path, 'rb') as f:
                    webp_data = webp.WebPData.from_buffer(f.read())
                    arr = webp_data.decode()
                    return Image.fromarray(arr)
            except Exception as e:
                logger.warning(f"WebP decode failed for {path}, falling back to PIL: {e}")
                return Image.open(path)
        else:
            return Image.open(path)
    
    def handle_transparency(self, img: Image.Image) -> Image.Image:
        """Handle transparency with gray background composite"""
        if img.mode in ('RGBA', 'LA'):
            # Create gray background
            background = Image.new('RGB', img.size, self.pad_color)
            # Composite with alpha channel
            if img.mode == 'RGBA':
                background.paste(img, mask=img.getchannel('A'))
            else:  # LA mode
                background.paste(img, mask=img.getchannel('L'))
            return background
        elif img.mode == 'P' and 'transparency' in img.info:
            # Handle palette images with transparency
            img = img.convert('RGBA')
            return self.handle_transparency(img)
        else:
            return img.convert('RGB')
    
    def letterbox_image(self, img: Image.Image, target_size: int) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """Letterbox image to target size, returns image and padding info"""
        w, h = img.size
        scale = min(target_size / w, target_size / h)
        
        # Resize maintaining aspect ratio
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create canvas with padding color
        canvas = Image.new('RGB', (target_size, target_size), self.pad_color)
        
        # Paste image centered
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        canvas.paste(img, (paste_x, paste_y))
        
        # Return padding info for potential region extraction later
        pad_info = (paste_x, paste_y, paste_x + new_w, paste_y + new_h)
        
        return canvas, pad_info
    
    def process_for_student(self, img: Image.Image) -> Tuple[np.ndarray, Dict]:
        """Process image as our student model will see it (640x640)"""
        # Handle transparency
        img = self.handle_transparency(img)
        original_size = img.size
        
        # Letterbox to 640x640
        img_letterboxed, pad_info = self.letterbox_image(img, self.student_size)
        
        # Convert to numpy array (uint8, no normalization yet)
        img_array = np.array(img_letterboxed)
        
        # Transpose to CHW format
        img_array = img_array.transpose(2, 0, 1)
        
        metadata = {
            'original_size': original_size,
            'pad_info': pad_info,  # (x1, y1, x2, y2) of actual image within letterbox
        }
        
        return img_array, metadata
    
    def process_for_anime_teacher(self, img: Image.Image) -> torch.Tensor:
        """Process image for anime teacher model (512x512)"""
        # Handle transparency
        img = self.handle_transparency(img)
        
        # Resize to 512x512 (what the model was trained on)
        img = img.resize((self.anime_teacher_size, self.anime_teacher_size), 
                        Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize
        tensor = T.ToTensor()(img)
        tensor = self.anime_normalize(tensor)
        
        return tensor
    
    def process_for_clip(self, img: Image.Image) -> torch.Tensor:
        """Process image for CLIP model (224x224 with CLIP preprocessing)"""
        # Handle transparency
        img = self.handle_transparency(img)
        
        # Use CLIP's processor - it handles resize to 224x224 and normalization
        inputs = self.clip_processor(images=img, return_tensors="pt")
        
        return inputs['pixel_values'][0]
    
    def process_image_all_models(self, path: Union[str, Path]) -> Dict:
        """Process image for all models at once"""
        try:
            # Load image once
            img = self.load_image(path)
            
            # Process for each model
            student_img, metadata = self.process_for_student(img)
            anime_teacher_img = self.process_for_anime_teacher(img)
            clip_img = self.process_for_clip(img)
            
            return {
                'student_image': student_img,
                'anime_teacher_image': anime_teacher_img,
                'clip_image': clip_img,
                'metadata': metadata,
                'path': str(path),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            return {
                'student_image': np.zeros((3, self.student_size, self.student_size), dtype=np.uint8),
                'anime_teacher_image': torch.zeros(3, self.anime_teacher_size, self.anime_teacher_size),
                'clip_image': torch.zeros(3, 224, 224),
                'metadata': {'error': str(e), 'path': str(path)},
                'path': str(path),
                'success': False
            }


class AnimeTeacherModel:
    """Wrapper for the anime-specific teacher model"""
    
    def __init__(self, model_path: str, device: int = 0):
        self.device = torch.device(f'cuda:{device}')
        self.model_path = model_path
        
        # Load model 
        logger.info(f"Loading anime teacher from {model_path}")
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)
        
        # Use automatic mixed precision
        self.use_amp = True
        
    def _load_model(self):
        """Load the anime teacher model - adjust based on actual format"""
        # This is a placeholder - adjust based on your actual model format
        # Could be .pth, .safetensors, etc.
        
        # Example for safetensors format:
        model_file = Path(self.model_path) / "model.safetensors"
        config_file = Path(self.model_path) / "config.json"
        
        if model_file.exists() and config_file.exists():
            # Load config
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Initialize model based on config
            # model = YourModelClass(config)
            
            # Load weights
            # state_dict = {}
            # with safe_open(model_file, framework="pt") as f:
            #     for key in f.keys():
            #         state_dict[key] = f.get_tensor(key)
            # model.load_state_dict(state_dict)
            pass
        
        # Placeholder - replace with actual loading
        logger.warning("Using placeholder model - implement actual loading!")
        import torch.nn as nn
        
        class DummyAnimeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(64, 70000)
                )
            
            def forward(self, x):
                return self.features(x)
        
        return DummyAnimeModel()
    
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from batch of images"""
        images = images.to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # Get model outputs
            outputs = self.model(images)
            
            # Extract different feature types
            features = {
                'tag_logits': outputs,  # Raw logits
                'tag_probs': torch.sigmoid(outputs),  # Probabilities
            }
            
            # Get top-k predictions
            k = min(100, outputs.shape[-1])
            top_k_probs, top_k_indices = torch.topk(features['tag_probs'], k=k, dim=-1)
            
            features['top_k_indices'] = top_k_indices
            features['top_k_probs'] = top_k_probs
        
        # Move to CPU for storage
        return {k: v.cpu().half() for k, v in features.items()}


class CLIPTeacherModel:
    """Wrapper for CLIP teacher model"""
    
    def __init__(self, model_id: str, device: int = 1):
        self.device = torch.device(f'cuda:{device}')
        self.model_id = model_id
        
        logger.info(f"Loading CLIP model {model_id}")
        
        # Load CLIP model
        from transformers import CLIPModel
        self.model = CLIPModel.from_pretrained(model_id)
        
        self.model.eval()
        self.model.to(self.device)
        self.model.half()  # Use fp16
        
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract CLIP features"""
        images = images.to(self.device)
        
        with torch.cuda.amp.autocast():
            # Get vision features
            vision_outputs = self.model.vision_model(pixel_values=images)
            
            # Extract different representations
            features = {
                'cls_token': vision_outputs.last_hidden_state[:, 0, :],  # CLS token
                'patch_tokens': vision_outputs.last_hidden_state[:, 1:, :],  # All patches
                'pooled': vision_outputs.pooler_output if hasattr(vision_outputs, 'pooler_output') else None
            }
            
            # Average pool patch tokens
            if features['patch_tokens'] is not None:
                features['patch_tokens_avg'] = features['patch_tokens'].mean(dim=1)
                
                # Get top-k patches by L2 norm
                patch_norms = features['patch_tokens'].norm(dim=-1)
                top_k_values, top_k_indices = torch.topk(patch_norms, k=min(16, patch_norms.shape[1]), dim=1)
                
                # Gather top patches
                batch_size = features['patch_tokens'].shape[0]
                batch_indices = torch.arange(batch_size).unsqueeze(1).expand_as(top_k_indices)
                features['top_patches'] = features['patch_tokens'][batch_indices, top_k_indices]
                features['top_patches_indices'] = top_k_indices
        
        # Move to CPU and convert to fp16
        return {k: v.cpu().half() if v is not None else None for k, v in features.items()}


class FeatureExtractionDataset(Dataset):
    """Dataset for loading images from multiple locations"""
    
    def __init__(self, 
                 file_list: List[Path],
                 preprocessor: ImagePreprocessor):
        self.files = file_list
        self.preprocessor = preprocessor
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        result = self.preprocessor.process_image_all_models(path)
        result['index'] = idx
        return result


class HDF5FeatureWriter:
    """Manages writing features to HDF5 files"""
    
    def __init__(self, output_dir: Path, chunk_size: int = 10000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        
        self.current_file = None
        self.current_h5 = None
        self.current_idx = 0
        self.file_counter = 0
        
    def _create_new_file(self):
        """Create a new HDF5 file"""
        if self.current_h5 is not None:
            self.current_h5.close()
            
        self.file_counter += 1
        filename = self.output_dir / f"features_part_{self.file_counter:05d}.h5"
        logger.info(f"Creating new feature file: {filename}")
        
        self.current_file = filename
        self.current_h5 = h5py.File(filename, 'w')
        self.current_idx = 0
        
        # Create groups
        self.current_h5.create_group('images')
        self.current_h5.create_group('anime_teacher')
        self.current_h5.create_group('clip_teacher')
        self.current_h5.create_group('metadata')
        
    def write_batch(self, batch_data: List[Dict]):
        """Write a batch of features"""
        
        if self.current_h5 is None or self.current_idx >= self.chunk_size:
            self._create_new_file()
            
        batch_size = len(batch_data)
        start_idx = self.current_idx
        end_idx = start_idx + batch_size
        
        # Initialize datasets if needed
        if self.current_idx == 0:
            # Student images (640x640, uint8)
            self.current_h5.create_dataset(
                'images/student_640',
                shape=(self.chunk_size, 3, 640, 640),
                dtype=np.uint8,
                compression='lzf',
                chunks=(1, 3, 640, 640)  # Chunk by single images
            )
            
            # Padding info
            self.current_h5.create_dataset(
                'images/padding_info',
                shape=(self.chunk_size, 4),
                dtype=np.int16
            )
            
            # Original sizes
            self.current_h5.create_dataset(
                'images/original_sizes',
                shape=(self.chunk_size, 2),
                dtype=np.int32
            )
            
            # Get shapes from first batch for feature initialization
            first_anime_features = None
            first_clip_features = None
            
            # Process first valid batch item to get shapes
            for item in batch_data:
                if item['success']:
                    # Extract features for shape inference
                    anime_imgs = torch.stack([torch.from_numpy(d['anime_teacher_image']) 
                                            for d in batch_data if d['success']])
                    clip_imgs = torch.stack([torch.from_numpy(d['clip_image']) 
                                           for d in batch_data if d['success']])
                    
                    if len(anime_imgs) > 0:
                        with torch.no_grad():
                            # Dummy extraction to get shapes
                            anime_teacher = globals().get('anime_teacher')
                            clip_teacher = globals().get('clip_teacher')
                            
                            if anime_teacher:
                                first_anime_features = anime_teacher.extract_features(anime_imgs[:1])
                            if clip_teacher:
                                first_clip_features = clip_teacher.extract_features(clip_imgs[:1])
                    break
            
            # Create anime teacher datasets
            if first_anime_features:
                for key, value in first_anime_features.items():
                    if value is not None:
                        shape = (self.chunk_size,) + value.shape[1:]
                        self.current_h5.create_dataset(
                            f'anime_teacher/{key}',
                            shape=shape,
                            dtype=np.float16,
                            compression='lzf'
                        )
            
            # Create CLIP datasets
            if first_clip_features:
                for key, value in first_clip_features.items():
                    if value is not None:
                        shape = (self.chunk_size,) + value.shape[1:]
                        self.current_h5.create_dataset(
                            f'clip_teacher/{key}',
                            shape=shape,
                            dtype=np.float16,
                            compression='lzf'
                        )
            
            # Metadata
            self.current_h5.create_dataset(
                'metadata/indices',
                shape=(self.chunk_size,),
                dtype=np.int64
            )
            
            # Store paths as variable-length strings
            dt = h5py.special_dtype(vlen=str)
            self.current_h5.create_dataset(
                'metadata/paths',
                shape=(self.chunk_size,),
                dtype=dt
            )
            
            self.current_h5.create_dataset(
                'metadata/success',
                shape=(self.chunk_size,),
                dtype=bool
            )
        
        # Write batch data
        for i, data in enumerate(batch_data):
            idx = start_idx + i
            
            # Write student image
            self.current_h5['images/student_640'][idx] = data['student_image']
            
            # Write metadata
            if data['success']:
                pad_info = data['metadata']['pad_info']
                self.current_h5['images/padding_info'][idx] = pad_info
                self.current_h5['images/original_sizes'][idx] = data['metadata']['original_size']
            
            self.current_h5['metadata/indices'][idx] = data['index']
            self.current_h5['metadata/paths'][idx] = data['path']
            self.current_h5['metadata/success'][idx] = data['success']
        
        self.current_idx = end_idx
        
        # Flush periodically
        if self.current_idx % 1000 == 0:
            self.current_h5.flush()
    
    def write_teacher_features(self, indices: List[int], anime_features: Dict, clip_features: Dict):
        """Write teacher features for specific indices"""
        if self.current_h5 is None:
            return
            
        # Map indices to positions in current file
        for i, idx in enumerate(indices):
            file_idx = idx % self.chunk_size
            
            # Write anime features
            for key, value in anime_features.items():
                if value is not None and f'anime_teacher/{key}' in self.current_h5:
                    self.current_h5[f'anime_teacher/{key}'][file_idx] = value[i].numpy()
            
            # Write CLIP features
            for key, value in clip_features.items():
                if value is not None and f'clip_teacher/{key}' in self.current_h5:
                    self.current_h5[f'clip_teacher/{key}'][file_idx] = value[i].numpy()
    
    def close(self):
        """Close current file"""
        if self.current_h5 is not None:
            # Resize datasets to actual size if needed
            if self.current_idx < self.chunk_size:
                for name in self.current_h5:
                    grp = self.current_h5[name]
                    for key in grp:
                        dataset = grp[key]
                        dataset.resize(self.current_idx, axis=0)
            
            self.current_h5.close()


def collate_batch(batch: List[Dict]) -> Dict:
    """Custom collate function for dataloader"""
    # Separate successful and failed items
    successful = [item for item in batch if item['success']]
    
    if not successful:
        return {
            'batch_data': batch,
            'anime_teacher_images': None,
            'clip_images': None,
            'indices': [item['index'] for item in batch]
        }
    
    # Stack teacher inputs
    anime_teacher_images = torch.stack([item['anime_teacher_image'] for item in successful])
    clip_images = torch.stack([item['clip_image'] for item in successful])
    
    return {
        'batch_data': batch,
        'anime_teacher_images': anime_teacher_images,
        'clip_images': clip_images,
        'indices': [item['index'] for item in successful]
    }


def discover_images(storage_locations: List[Dict]) -> List[Path]:
    """Discover all images across storage locations"""
    logger.info("Discovering images across storage locations...")
    
    all_files = []
    supported_extensions = {'.webp', '.jpg', '.jpeg', '.png'}
    
    def scan_directory(path: Path) -> List[Path]:
        """Recursively scan directory for images"""
        images = []
        for ext in supported_extensions:
            images.extend(path.rglob(f'*{ext}'))
        return images
    
    # Scan each location in parallel
    with ProcessPoolExecutor(max_workers=len(storage_locations)) as executor:
        futures = []
        for location in storage_locations:
            path = Path(location['path'])
            if path.exists():
                logger.info(f"Scanning {path} (priority {location['priority']})...")
                future = executor.submit(scan_directory, path)
                futures.append((future, location))
            else:
                logger.warning(f"Path does not exist: {path}")
        
        # Collect results
        for future, location in futures:
            try:
                files = future.result()
                logger.info(f"Found {len(files)} images in {location['path']}")
                all_files.extend(files)
            except Exception as e:
                logger.error(f"Error scanning {location['path']}: {e}")
    
    # Sort by priority (lower number = higher priority)
    location_map = {loc['path']: loc['priority'] for loc in storage_locations}
    all_files.sort(key=lambda f: location_map.get(str(f.parent), 999))
    
    logger.info(f"Total images discovered: {len(all_files)}")
    return all_files


def deduplicate_files(file_list: List[Path]) -> List[Path]:
    """Basic file-level deduplication using hash"""
    logger.info("Deduplicating files...")
    
    seen_hashes = set()
    unique_files = []
    
    def get_file_hash(path: Path) -> str:
        """Get xxhash of file"""
        import xxhash
        h = xxhash.xxh64()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
        return h.hexdigest()
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=32) as executor:
        # Submit all hash jobs
        future_to_path = {executor.submit(get_file_hash, path): path for path in file_list}
        
        for future in tqdm(future_to_path, desc="Computing file hashes"):
            try:
                path = future_to_path[future]
                file_hash = future.result()
                
                if file_hash not in seen_hashes:
                    seen_hashes.add(file_hash)
                    unique_files.append(path)
            except Exception as e:
                logger.error(f"Error hashing file: {e}")
    
    logger.info(f"Deduplication complete: {len(unique_files)}/{len(file_list)} unique files")
    return unique_files


def main():
    """Main preprocessing pipeline"""
    config = PreprocessConfig()
    
    # Setup
    logger.info("Initializing preprocessing pipeline...")
    preprocessor = ImagePreprocessor(config)
    
    # Initialize teachers
    logger.info("Loading teacher models...")
    anime_teacher = AnimeTeacherModel(
        config.anime_teacher_path,
        device=config.anime_teacher_gpu
    )
    
    clip_teacher = CLIPTeacherModel(
        config.clip_model_id,
        device=config.clip_teacher_gpu
    )
    
    # Make teachers globally accessible for writer
    globals()['anime_teacher'] = anime_teacher
    globals()['clip_teacher'] = clip_teacher
    
    # Discover and deduplicate images
    all_files = discover_images(config.storage_locations)
    unique_files = deduplicate_files(all_files)
    
    # Create dataset
    dataset = FeatureExtractionDataset(unique_files, preprocessor)
    
    # Single dataloader that prepares images for all models
    dataloader = DataLoader(
        dataset,
        batch_size=32,  # Smaller batch since we process for all models
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=collate_batch,
        drop_last=False
    )
    
    # Setup feature writer
    writer = HDF5FeatureWriter(
        output_dir=Path(config.output_dir),
        chunk_size=config.chunk_size
    )
    
    # Process batches
    logger.info("Starting feature extraction...")
    
    try:
        for batch in tqdm(dataloader, desc="Extracting features"):
            # Write student images and metadata
            writer.write_batch(batch['batch_data'])
            
            # Extract and write teacher features if we have valid images
            if batch['anime_teacher_images'] is not None:
                # Extract anime teacher features
                anime_features = anime_teacher.extract_features(batch['anime_teacher_images'])
                
                # Extract CLIP features  
                clip_features = clip_teacher.extract_features(batch['clip_images'])
                
                # Write teacher features
                writer.write_teacher_features(
                    batch['indices'],
                    anime_features,
                    clip_features
                )
                
    finally:
        writer.close()
        logger.info("Feature extraction complete!")
        
    # Create index file
    index_file = Path(config.output_dir) / "index.json"
    index_data = {
        'total_images': len(unique_files),
        'chunk_size': config.chunk_size,
        'num_files': writer.file_counter,
        'file_paths': [str(f) for f in unique_files[:1000]],  # Just store first 1000 as sample
        'config': {
            'student_image_size': config.student_image_size,
            'anime_teacher_size': config.anime_teacher_size,
            'anime_teacher_path': config.anime_teacher_path,
            'clip_model_id': config.clip_model_id
        }
    }
    
    with open(index_file, 'w') as f:
        json.dump(index_data, f, indent=2)
    
    logger.info(f"Index file written to {index_file}")


if __name__ == "__main__":
    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Run preprocessing
    main()
