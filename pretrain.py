#!/usr/bin/env python3
"""
Anime Image Tagger - Teacher Feature Extraction Pipeline
Preprocesses images and extracts features from dual teachers for training
Correctly handles different preprocessing requirements for each model
"""

import os
import sys
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import gc

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import h5py
from tqdm import tqdm
import transformers
from safetensors import safe_open
from safetensors.torch import load_file

# Configure PIL to handle large images and WebP
Image.MAX_IMAGE_PIXELS = None
try:
    from PIL import WebPImagePlugin
    WEBP_AVAILABLE = True
except ImportError:
    WEBP_AVAILABLE = False
    print("Warning: WebP support not available. Install pillow-webp-plugin for WebP support.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing pipeline"""
    # Model paths
    anime_teacher_path: str = "/models/anime_tag_model_70k"
    clip_model_id: str = "zer0int/CLIP-SAE-ViT-L-14"
    
    # Model-specific image sizes
    student_image_size: int = 640  # Our model
    anime_teacher_size: int = 512  # Anime model trained on 512
    # CLIP size handled by processor (224)
    
    # Processing settings
    batch_size: int = 32  # Unified batch size
    num_workers: int = 8
    
    # Output settings
    output_dir: str = "/home/user/datasets/teacher_features"
    chunk_size: int = 10000
    
    # GPU allocation
    anime_teacher_gpu: int = 0
    clip_teacher_gpu: int = 1
    
    # Gray padding color
    pad_color: Tuple[int, int, int] = (114, 114, 114)
    
    # Storage locations in priority order
    storage_locations: List[Dict[str, Union[str, int]]] = field(default_factory=lambda: [
        {"path": "/home/user/datasets/anime_curated", "priority": 0, "type": "local"},
        {"path": "/mnt/das/anime_archive", "priority": 1, "type": "das"},
        {"path": "/mnt/nas/anime_dataset/primary", "priority": 2, "type": "nas"},
        {"path": "/mnt/nas/anime_dataset/video_frames", "priority": 3, "type": "nas"}
    ])


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
        try:
            from transformers import CLIPProcessor
            self.clip_processor = CLIPProcessor.from_pretrained(config.clip_model_id)
        except Exception as e:
            logger.error(f"Failed to load CLIP processor: {e}")
            raise
    
    def load_image(self, path: Union[str, Path]) -> Image.Image:
        """Load image handling WebP and other formats"""
        path = Path(path)
        
        try:
            # PIL should handle WebP if plugin is installed
            img = Image.open(path)
            # Convert to RGB if needed (handles various modes)
            if img.mode not in ('RGB', 'RGBA', 'L', 'LA', 'P'):
                img = img.convert('RGB')
            return img
        except Exception as e:
            logger.error(f"Failed to load image {path}: {e}")
            raise
    
    def handle_transparency(self, img: Image.Image) -> Image.Image:
        """Handle transparency with gray background composite"""
        if img.mode in ('RGBA', 'LA'):
            # Create gray background
            background = Image.new('RGB', img.size, self.pad_color)
            # Composite with alpha channel
            if img.mode == 'RGBA':
                background.paste(img, mask=img.split()[3])  # Use alpha channel
            else:  # LA mode
                background.paste(img, mask=img.split()[1])  # Use alpha channel
            return background
        elif img.mode == 'P':
            # Convert palette images to RGBA first if they have transparency
            if 'transparency' in img.info:
                img = img.convert('RGBA')
                return self.handle_transparency(img)
            else:
                return img.convert('RGB')
        elif img.mode == 'L':
            # Convert grayscale to RGB
            return img.convert('RGB')
        else:
            return img.convert('RGB') if img.mode != 'RGB' else img
    
    def letterbox_image(self, img: Image.Image, target_size: int) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """Letterbox image to target size, returns image and padding info"""
        w, h = img.size
        scale = min(target_size / w, target_size / h)
        
        # Resize maintaining aspect ratio
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Ensure dimensions are at least 1
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        
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
        img_array = np.array(img_letterboxed, dtype=np.uint8)
        
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
                'metadata': {'error': str(e), 'path': str(path), 'original_size': (0, 0), 'pad_info': (0, 0, 0, 0)},
                'path': str(path),
                'success': False
            }


class AnimeTeacherModel:
    """Wrapper for the anime-specific teacher model"""
    
    def __init__(self, model_path: str, device: int = 0):
        self.device = torch.device(f'cuda:{device}')
        self.model_path = Path(model_path)
        
        # Load model 
        logger.info(f"Loading anime teacher from {model_path}")
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)
        
        # Use automatic mixed precision
        self.use_amp = True
        
    def _load_model(self):
        """Load the anime teacher model"""
        model_file = self.model_path / "model.safetensors"
        config_file = self.model_path / "config.json"
        
        # Try safetensors format first
        if model_file.exists() and config_file.exists():
            try:
                # Load config
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Try to load using transformers
                model_type = config.get('model_type', 'vit')
                
                if 'vit' in model_type.lower():
                    from transformers import ViTForImageClassification, ViTConfig
                    model_config = ViTConfig(**config)
                    model = ViTForImageClassification(model_config)
                else:
                    # Fallback to timm
                    import timm
                    model = timm.create_model(
                        config.get('architecture', 'vit_large_patch14_clip_224'),
                        pretrained=False,
                        num_classes=config.get('num_classes', 70000)
                    )
                
                # Load weights
                state_dict = load_file(model_file)
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded model from {model_file}")
                return model
                
            except Exception as e:
                logger.warning(f"Failed to load from safetensors: {e}, trying PyTorch format")
        
        # Try PyTorch format
        pytorch_file = self.model_path / "pytorch_model.bin"
        if pytorch_file.exists():
            try:
                # Use timm as fallback
                import timm
                
                # Load checkpoint
                checkpoint = torch.load(pytorch_file, map_location='cpu')
                
                # Try to infer model architecture
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Create model (assuming ViT)
                model = timm.create_model(
                    'vit_large_patch14_clip_224',
                    pretrained=False,
                    num_classes=70000  # Adjust based on your needs
                )
                
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded PyTorch model from {pytorch_file}")
                return model
                
            except Exception as e:
                logger.error(f"Failed to load PyTorch model: {e}")
                raise
        
        raise FileNotFoundError(f"No valid model file found in {self.model_path}")
    
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from batch of images"""
        images = images.to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # Get model outputs
            outputs = self.model(images)
            
            # Handle different output types
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Extract different feature types
            features = {
                'tag_logits': logits.cpu().half(),  # Raw logits
                'tag_probs': torch.sigmoid(logits).cpu().half(),  # Probabilities
            }
            
            # Get top-k predictions
            k = min(100, logits.shape[-1])
            top_k_probs, top_k_indices = torch.topk(torch.sigmoid(logits), k=k, dim=-1)
            
            features['top_k_indices'] = top_k_indices.cpu()
            features['top_k_probs'] = top_k_probs.cpu().half()
        
        return features


class CLIPTeacherModel:
    """Wrapper for CLIP teacher model"""
    
    def __init__(self, model_id: str, device: int = 1):
        self.device = torch.device(f'cuda:{device}')
        self.model_id = model_id
        
        logger.info(f"Loading CLIP model {model_id}")
        
        try:
            from transformers import CLIPModel
            self.model = CLIPModel.from_pretrained(
                model_id, 
                torch_dtype=torch.float16,
                device_map={'': device}
            )
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
        
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract CLIP features"""
        images = images.to(self.device)
        
        with torch.cuda.amp.autocast():
            # Get vision features
            vision_outputs = self.model.vision_model(pixel_values=images)
            
            # Extract different representations
            features = {
                'cls_token': vision_outputs.last_hidden_state[:, 0, :].cpu().half(),  # CLS token
                'patch_tokens': vision_outputs.last_hidden_state[:, 1:, :].cpu().half(),  # All patches
            }
            
            # Get pooled output if available
            if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                features['pooled'] = vision_outputs.pooler_output.cpu().half()
            else:
                features['pooled'] = None
            
            # Average pool patch tokens
            features['patch_tokens_avg'] = features['patch_tokens'].mean(dim=1)
            
            # Get top-k patches by L2 norm
            patch_norms = features['patch_tokens'].norm(dim=-1)
            k = min(16, patch_norms.shape[1])
            top_k_values, top_k_indices = torch.topk(patch_norms, k=k, dim=1)
            
            # Gather top patches
            batch_size = features['patch_tokens'].shape[0]
            batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, k)
            features['top_patches'] = features['patch_tokens'][batch_indices, top_k_indices]
            features['top_patches_indices'] = top_k_indices
        
        return features


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
        self.global_idx = 0  # Track global index across files
        
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
        
    def write_batch(self, batch_data: List[Dict], anime_features: Dict = None, clip_features: Dict = None):
        """Write a batch of features"""
        
        if self.current_h5 is None or self.current_idx >= self.chunk_size:
            self._create_new_file()
            
        batch_size = len(batch_data)
        
        # Initialize datasets if needed
        if self.current_idx == 0:
            self._initialize_datasets(batch_data, anime_features, clip_features)
        
        # Process successful items
        success_idx = 0
        for i, data in enumerate(batch_data):
            idx = self.current_idx + i
            
            # Write student image
            self.current_h5['images/student_640'][idx] = data['student_image']
            
            # Write metadata
            self.current_h5['images/padding_info'][idx] = data['metadata']['pad_info']
            self.current_h5['images/original_sizes'][idx] = data['metadata']['original_size']
            
            self.current_h5['metadata/indices'][idx] = data['index']
            self.current_h5['metadata/paths'][idx] = data['path']
            self.current_h5['metadata/success'][idx] = data['success']
            
            # Write teacher features for successful items
            if data['success'] and anime_features is not None and clip_features is not None:
                # Write anime features
                for key, value in anime_features.items():
                    if value is not None and f'anime_teacher/{key}' in self.current_h5:
                        self.current_h5[f'anime_teacher/{key}'][idx] = value[success_idx].numpy()
                
                # Write CLIP features
                for key, value in clip_features.items():
                    if value is not None and f'clip_teacher/{key}' in self.current_h5:
                        self.current_h5[f'clip_teacher/{key}'][idx] = value[success_idx].numpy()
                
                success_idx += 1
        
        self.current_idx += batch_size
        self.global_idx += batch_size
        
        # Flush periodically
        if self.current_idx % 1000 == 0:
            self.current_h5.flush()
    
    def _initialize_datasets(self, batch_data: List[Dict], anime_features: Dict = None, clip_features: Dict = None):
        """Initialize HDF5 datasets based on first batch"""
        # Student images (640x640, uint8)
        self.current_h5.create_dataset(
            'images/student_640',
            shape=(self.chunk_size, 3, 640, 640),
            dtype=np.uint8,
            compression='lzf',
            chunks=(1, 3, 640, 640)
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
        
        # Create teacher datasets if features provided
        if anime_features is not None:
            for key, value in anime_features.items():
                if value is not None:
                    shape = (self.chunk_size,) + value.shape[1:]
                    self.current_h5.create_dataset(
                        f'anime_teacher/{key}',
                        shape=shape,
                        dtype=np.float16,
                        compression='lzf'
                    )
        
        if clip_features is not None:
            for key, value in clip_features.items():
                if value is not None:
                    shape = (self.chunk_size,) + value.shape[1:]
                    self.current_h5.create_dataset(
                        f'clip_teacher/{key}',
                        shape=shape,
                        dtype=np.float16,
                        compression='lzf'
                    )
    
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
            'clip_images': None
        }
    
    # Stack teacher inputs
    anime_teacher_images = torch.stack([item['anime_teacher_image'] for item in successful])
    clip_images = torch.stack([item['clip_image'] for item in successful])
    
    return {
        'batch_data': batch,
        'anime_teacher_images': anime_teacher_images,
        'clip_images': clip_images
    }


def discover_images(storage_locations: List[Dict]) -> List[Path]:
    """Discover all images across storage locations"""
    logger.info("Discovering images across storage locations...")
    
    all_files = []
    supported_extensions = {'.webp', '.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    
    for location in storage_locations:
        path = Path(location['path'])
        if path.exists():
            logger.info(f"Scanning {path} (priority {location['priority']})...")
            
            # Scan for each extension
            for ext in supported_extensions:
                files = list(path.rglob(f'*{ext}'))
                files.extend(list(path.rglob(f'*{ext.upper()}')))
                all_files.extend(files)
            
            logger.info(f"Found {len(all_files)} images so far")
        else:
            logger.warning(f"Path does not exist: {path}")
    
    # Remove duplicates
    all_files = list(set(all_files))
    
    # Sort by priority (lower number = higher priority)
    location_map = {Path(loc['path']): loc['priority'] for loc in storage_locations}
    
    def get_priority(file_path):
        for loc_path, priority in location_map.items():
            if loc_path in file_path.parents or loc_path == file_path.parent:
                return priority
        return 999
    
    all_files.sort(key=get_priority)
    
    logger.info(f"Total unique images discovered: {len(all_files)}")
    return all_files


def deduplicate_files(file_list: List[Path]) -> List[Path]:
    """Basic file-level deduplication using file size and hash"""
    logger.info("Deduplicating files...")
    
    seen_hashes = set()
    unique_files = []
    
    def get_file_hash(path: Path) -> Optional[str]:
        """Get hash of file"""
        try:
            # Use size + first/last bytes for quick hash
            stat = path.stat()
            size = stat.st_size
            
            # For small files, hash entire content
            if size < 1024 * 1024:  # 1MB
                with open(path, 'rb') as f:
                    return hashlib.md5(f.read()).hexdigest()
            
            # For larger files, hash size + samples
            hasher = hashlib.md5()
            hasher.update(str(size).encode())
            
            with open(path, 'rb') as f:
                # Hash first 64KB
                hasher.update(f.read(65536))
                # Hash last 64KB
                f.seek(-65536, 2)
                hasher.update(f.read(65536))
            
            return hasher.hexdigest()
            
        except Exception as e:
            logger.error(f"Error hashing {path}: {e}")
            return None
    
    # Process files
    for path in tqdm(file_list, desc="Computing file hashes"):
        file_hash = get_file_hash(path)
        
        if file_hash and file_hash not in seen_hashes:
            seen_hashes.add(file_hash)
            unique_files.append(path)
    
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
    
    # Discover and deduplicate images
    all_files = discover_images(config.storage_locations)
    unique_files = deduplicate_files(all_files)
    
    if not unique_files:
        logger.error("No images found!")
        return
    
    # Create dataset
    dataset = FeatureExtractionDataset(unique_files, preprocessor)
    
    # Single dataloader that prepares images for all models
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if config.num_workers > 0 else False,
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
            # Extract teacher features if we have valid images
            anime_features = None
            clip_features = None
            
            if batch['anime_teacher_images'] is not None:
                # Extract anime teacher features
                anime_features = anime_teacher.extract_features(batch['anime_teacher_images'])
                
                # Extract CLIP features  
                clip_features = clip_teacher.extract_features(batch['clip_images'])
            
            # Write batch with features
            writer.write_batch(batch['batch_data'], anime_features, clip_features)
            
            # Clear GPU cache periodically
            if writer.global_idx % 1000 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise
    finally:
        writer.close()
        logger.info("Feature extraction complete!")
        
    # Create index file
    index_file = Path(config.output_dir) / "index.json"
    index_data = {
        'total_images': len(unique_files),
        'chunk_size': config.chunk_size,
        'num_files': writer.file_counter,
        'total_processed': writer.global_idx,
        'file_paths': [str(f) for f in unique_files[:1000]],  # Sample of paths
        'config': {
            'student_image_size': config.student_image_size,
            'anime_teacher_size': config.anime_teacher_size,
            'anime_teacher_path': config.anime_teacher_path,
            'clip_model_id': config.clip_model_id,
            'batch_size': config.batch_size,
            'num_workers': config.num_workers
        }
    }
    
    with open(index_file, 'w') as f:
        json.dump(index_data, f, indent=2)
    
    logger.info(f"Index file written to {index_file}")
    logger.info(f"Processed {writer.global_idx} images into {writer.file_counter} files")


if __name__ == "__main__":
    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Run preprocessing
    main()