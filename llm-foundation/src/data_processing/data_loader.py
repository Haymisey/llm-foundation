"""
PyTorch DataLoader for Amharic-Oromiffa Translation

This module creates PyTorch DataLoaders for training the transformer
model with the tokenized parallel text data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Iterator
import logging
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationDataset(Dataset):
    """
    PyTorch Dataset for Amharic-Oromiffa translation pairs
    """
    
    def __init__(self, data_pairs: List[Tuple[str, str]], 
                 vocabulary, max_length: Optional[int] = None):
        """
        Initialize translation dataset
        
        Args:
            data_pairs: List of (amharic_text, oromiffa_text) pairs
            vocabulary: Vocabulary manager instance
            max_length: Maximum sequence length
        """
        self.data_pairs = data_pairs
        self.vocabulary = vocabulary
        self.max_length = max_length
        
        # Pre-tokenize all data for efficiency
        logger.info(f"🔧 Pre-tokenizing {len(data_pairs)} translation pairs...")
        self.tokenized_data = []
        
        for i, (amh_text, omo_text) in enumerate(data_pairs):
            try:
                # Encode the pair
                amh_ids, omo_ids = self.vocabulary.encode_pair(amh_text, omo_text, max_length)
                
                # Store tokenized data
                self.tokenized_data.append({
                    'amharic_ids': amh_ids,
                    'oromiffa_ids': omo_ids,
                    'amharic_length': len(amh_ids),
                    'oromiffa_length': len(omo_ids)
                })
                
                # Progress logging
                if (i + 1) % 1000 == 0:
                    logger.info(f"   Processed {i + 1}/{len(data_pairs)} pairs...")
                    
            except Exception as e:
                logger.warning(f"Error processing pair {i}: {e}")
                continue
        
        logger.info(f"✅ Successfully tokenized {len(self.tokenized_data)} pairs")
        
        # Calculate sequence length statistics
        self._log_length_stats()
    
    def _log_length_stats(self):
        """Log sequence length statistics"""
        if not self.tokenized_data:
            return
        
        amh_lengths = [item['amharic_length'] for item in self.tokenized_data]
        omo_lengths = [item['oromiffa_length'] for item in self.tokenized_data]
        
        logger.info("📊 Tokenized Data Statistics:")
        logger.info(f"   Amharic - Avg length: {np.mean(amh_lengths):.1f} tokens")
        logger.info(f"   Amharic - Min length: {min(amh_lengths)} tokens")
        logger.info(f"   Amharic - Max length: {max(amh_lengths)} tokens")
        logger.info(f"   Oromiffa - Avg length: {np.mean(omo_lengths):.1f} tokens")
        logger.info(f"   Oromiffa - Min length: {min(omo_lengths)} tokens")
        logger.info(f"   Oromiffa - Max length: {max(omo_lengths)} tokens")
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.tokenized_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single translation pair
        
        Args:
            idx: Index of the pair
            
        Returns:
            Dictionary with tokenized data
        """
        item = self.tokenized_data[idx]
        
        return {
            'amharic_ids': torch.tensor(item['amharic_ids'], dtype=torch.long),
            'oromiffa_ids': torch.tensor(item['oromiffa_ids'], dtype=torch.long),
            'amharic_length': item['amharic_length'],
            'oromiffa_length': item['oromiffa_length']
        }

class TranslationDataLoader:
    """
    DataLoader manager for translation training
    """
    
    def __init__(self, vocabulary, batch_size: int = 32, 
                 max_length: Optional[int] = None,
                 shuffle: bool = True, num_workers: int = 0):
        """
        Initialize data loader manager
        
        Args:
            vocabulary: Vocabulary manager instance
            batch_size: Batch size for training
            max_length: Maximum sequence length
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
        """
        self.vocabulary = vocabulary
        self.batch_size = batch_size
        self.max_length = max_length
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # DataLoaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        logger.info(f"📚 Initializing TranslationDataLoader with batch_size={batch_size}")
    
    def create_datasets(self, data_splits: Dict[str, List[Tuple[str, str]]]):
        """
        Create PyTorch datasets from data splits
        
        Args:
            data_splits: Dictionary with 'train', 'val', 'test' splits
        """
        logger.info("🔧 Creating PyTorch datasets...")
        
        # Create training dataset
        if 'train' in data_splits:
            self.train_dataset = TranslationDataset(
                data_splits['train'], 
                self.vocabulary, 
                self.max_length
            )
            logger.info(f"✅ Training dataset created: {len(self.train_dataset)} samples")
        
        # Create validation dataset
        if 'val' in data_splits:
            self.val_dataset = TranslationDataset(
                data_splits['val'], 
                self.vocabulary, 
                self.max_length
            )
            logger.info(f"✅ Validation dataset created: {len(self.val_dataset)} samples")
        
        # Create test dataset
        if 'test' in data_splits:
            self.test_dataset = TranslationDataset(
                data_splits['test'], 
                self.vocabulary, 
                self.max_length
            )
            logger.info(f"✅ Test dataset created: {len(self.test_dataset)} samples")
    
    def create_data_loaders(self):
        """Create PyTorch DataLoaders from datasets"""
        logger.info("🔧 Creating PyTorch DataLoaders...")
        
        # Training DataLoader
        if self.train_dataset:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                collate_fn=self._collate_fn,
                drop_last=True  # Drop incomplete batches
            )
            logger.info(f"✅ Training DataLoader created: {len(self.train_loader)} batches")
        
        # Validation DataLoader
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,  # No shuffling for validation
                num_workers=self.num_workers,
                collate_fn=self._collate_fn,
                drop_last=False  # Keep all validation data
            )
            logger.info(f"✅ Validation DataLoader created: {len(self.val_loader)} batches")
        
        # Test DataLoader
        if self.test_dataset:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,  # No shuffling for testing
                num_workers=self.num_workers,
                collate_fn=self._collate_fn,
                drop_last=False  # Keep all test data
            )
            logger.info(f"✅ Test DataLoader created: {len(self.test_loader)} batches")
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function to handle variable-length sequences
        
        Args:
            batch: List of dataset items
            
        Returns:
            Batched tensor data
        """
        # Extract data from batch
        amharic_ids = [item['amharic_ids'] for item in batch]
        oromiffa_ids = [item['oromiffa_ids'] for item in batch]
        
        # Pad sequences to the same length within the batch
        padded_amharic = self.vocabulary.pad_sequences(amharic_ids)
        padded_oromiffa = self.vocabulary.pad_sequences(oromiffa_ids)
        
        # Create attention masks
        amharic_mask = self.vocabulary.create_attention_mask(padded_amharic)
        oromiffa_mask = self.vocabulary.create_attention_mask(padded_oromiffa)
        
        # Convert to tensors
        return {
            'amharic_ids': torch.tensor(padded_amharic, dtype=torch.long),
            'oromiffa_ids': torch.tensor(padded_oromiffa, dtype=torch.long),
            'amharic_mask': torch.tensor(amharic_mask, dtype=torch.long),
            'oromiffa_mask': torch.tensor(oromiffa_mask, dtype=torch.long),
            'batch_size': len(batch)
        }
    
    def get_data_loaders(self) -> Dict[str, DataLoader]:
        """
        Get all created DataLoaders
        
        Returns:
            Dictionary with 'train', 'val', 'test' DataLoaders
        """
        loaders = {}
        
        if self.train_loader:
            loaders['train'] = self.train_loader
        if self.val_loader:
            loaders['val'] = self.val_loader
        if self.test_loader:
            loaders['test'] = self.test_loader
        
        return loaders
    
    def get_batch_info(self) -> Dict[str, Dict[str, int]]:
        """
        Get information about the created batches
        
        Returns:
            Dictionary with batch information
        """
        info = {}
        
        for split_name, loader in self.get_data_loaders().items():
            if loader:
                # Get a sample batch to determine shapes
                sample_batch = next(iter(loader))
                
                info[split_name] = {
                    'num_batches': len(loader),
                    'batch_size': sample_batch['batch_size'],
                    'amharic_shape': sample_batch['amharic_ids'].shape,
                    'oromiffa_shape': sample_batch['oromiffa_ids'].shape,
                    'amharic_mask_shape': sample_batch['amharic_mask'].shape,
                    'oromiffa_mask_shape': sample_batch['oromiffa_mask'].shape
                }
        
        return info
    
    def test_data_loader(self, num_batches: int = 2):
        """
        Test the data loader with a few batches
        
        Args:
            num_batches: Number of batches to test
        """
        if not self.train_loader:
            logger.error("No training DataLoader available. Create datasets first.")
            return
        
        logger.info("🧪 Testing DataLoader with sample batches...")
        
        for i, batch in enumerate(self.train_loader):
            if i >= num_batches:
                break
            
            logger.info(f"\n   Batch {i + 1}:")
            logger.info(f"      Batch size: {batch['batch_size']}")
            logger.info(f"      Amharic shape: {batch['amharic_ids'].shape}")
            logger.info(f"      Oromiffa shape: {batch['oromiffa_ids'].shape}")
            logger.info(f"      Amharic mask shape: {batch['amharic_mask'].shape}")
            logger.info(f"      Oromiffa mask shape: {batch['oromiffa_mask'].shape}")
            
            # Show sample tokens
            sample_amh = batch['amharic_ids'][0]
            sample_omo = batch['oromiffa_ids'][0]
            
            logger.info(f"      Sample Amharic IDs: {sample_amh[:10].tolist()}...")
            logger.info(f"      Sample Oromiffa IDs: {sample_omo[:10].tolist()}...")
            
            # Decode sample
            try:
                amh_text = self.vocabulary.decode_amharic(sample_amh.tolist())
                omo_text = self.vocabulary.decode_oromiffa(sample_omo.tolist())
                logger.info(f"      Sample Amharic: '{amh_text[:50]}...'")
                logger.info(f"      Sample Oromiffa: '{omo_text[:50]}...'")
            except Exception as e:
                logger.warning(f"      Error decoding sample: {e}")
    
    def save_data_loader_info(self, save_path: str = "data_loader_info.json"):
        """
        Save DataLoader information to file
        
        Args:
            save_path: Path to save DataLoader info
        """
        import json
        
        info = {
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'shuffle': self.shuffle,
            'num_workers': self.num_workers,
            'batch_info': self.get_batch_info(),
            'vocabulary_stats': self.vocabulary.get_vocabulary_stats()
        }
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 DataLoader info saved to {save_path}")
