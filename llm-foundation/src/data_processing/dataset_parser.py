"""
Dataset Parser for Amharic-Oromiffa Translation Data

This module handles parsing the tab-separated parallel text file
and preparing it for tokenization and training.
"""

import os
import pandas as pd
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AmharicOromiffaDataset:
    """
    Dataset parser for Amharic-Oromiffa parallel text
    """
    
    def __init__(self, data_path: str = "data/amh_omo.txt"):
        """
        Initialize dataset parser
        
        Args:
            data_path: Path to the tab-separated parallel text file
        """
        self.data_path = Path(data_path)
        self.amharic_texts = []
        self.oromiffa_texts = []
        self.raw_data = []
        
        # Validate file exists
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        logger.info(f"Initializing dataset parser for: {self.data_path}")
    
    def parse_data(self) -> Tuple[List[str], List[str]]:
        """
        Parse the tab-separated parallel text file
        
        Returns:
            Tuple of (amharic_texts, oromiffa_texts)
        """
        logger.info("Parsing parallel text data...")
        
        try:
            # Read the file line by line to handle large files efficiently
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    # Split by tab character
                    parts = line.split('\t')
                    if len(parts) != 2:
                        logger.warning(f"Line {line_num}: Invalid format (expected 2 parts, got {len(parts)})")
                        logger.warning(f"Line content: {line[:100]}...")
                        continue
                    
                    amharic, oromiffa = parts[0].strip(), parts[1].strip()
                    
                    # Basic validation
                    if amharic and oromiffa:  # Both texts should be non-empty
                        self.amharic_texts.append(amharic)
                        self.oromiffa_texts.append(oromiffa)
                        self.raw_data.append((amharic, oromiffa))
                    
                    # Progress logging for large files
                    if line_num % 10000 == 0:
                        logger.info(f"Processed {line_num} lines...")
            
            logger.info(f"✅ Successfully parsed {len(self.amharic_texts)} parallel text pairs")
            
            # Log some statistics
            self._log_dataset_stats()
            
            return self.amharic_texts, self.oromiffa_texts
            
        except Exception as e:
            logger.error(f"Error parsing data file: {e}")
            raise
    
    def _log_dataset_stats(self):
        """Log dataset statistics"""
        if not self.amharic_texts:
            return
            
        # Calculate statistics
        amharic_lengths = [len(text) for text in self.amharic_texts]
        oromiffa_lengths = [len(text) for text in self.oromiffa_texts]
        
        logger.info("📊 Dataset Statistics:")
        logger.info(f"   Total pairs: {len(self.amharic_texts)}")
        logger.info(f"   Amharic - Avg length: {sum(amharic_lengths)/len(amharic_lengths):.1f} chars")
        logger.info(f"   Amharic - Min length: {min(amharic_lengths)} chars")
        logger.info(f"   Amharic - Max length: {max(amharic_lengths)} chars")
        logger.info(f"   Oromiffa - Avg length: {sum(oromiffa_lengths)/len(oromiffa_lengths):.1f} chars")
        logger.info(f"   Oromiffa - Min length: {min(oromiffa_lengths)} chars")
        logger.info(f"   Oromiffa - Max length: {max(oromiffa_lengths)} chars")
        
        # Show sample pairs
        logger.info("\n📝 Sample pairs:")
        for i in range(min(5, len(self.raw_data))):
            amh, omo = self.raw_data[i]
            logger.info(f"   {i+1}. Amharic: {amh[:50]}{'...' if len(amh) > 50 else ''}")
            logger.info(f"      Oromiffa: {omo[:50]}{'...' if len(omo) > 50 else ''}")
    
    def get_training_data(self, train_split: float = 0.8, 
                         val_split: float = 0.1,
                         test_split: float = 0.1) -> Dict[str, List[Tuple[str, str]]]:
        """
        Split data into training, validation, and test sets
        
        Args:
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        if not self.raw_data:
            self.parse_data()
        
        # Validate splits
        total_split = train_split + val_split + test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Splits must sum to 1.0, got {total_split}")
        
        # Shuffle data
        import random
        random.shuffle(self.raw_data)
        
        # Calculate split indices
        total_samples = len(self.raw_data)
        train_end = int(total_samples * train_split)
        val_end = train_end + int(total_samples * val_split)
        
        # Split data
        train_data = self.raw_data[:train_end]
        val_data = self.raw_data[train_end:val_end]
        test_data = self.raw_data[val_end:]
        
        logger.info(f"📊 Data splits:")
        logger.info(f"   Training: {len(train_data)} pairs ({len(train_data)/total_samples*100:.1f}%)")
        logger.info(f"   Validation: {len(val_data)} pairs ({len(val_data)/total_samples*100:.1f}%)")
        logger.info(f"   Test: {len(test_data)} pairs ({len(test_data)/total_samples*100:.1f}%)")
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def save_splits(self, output_dir: str = "data/processed"):
        """
        Save data splits to separate files
        
        Args:
            output_dir: Directory to save split files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        splits = self.get_training_data()
        
        for split_name, split_data in splits.items():
            output_file = output_path / f"{split_name}.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for amh, omo in split_data:
                    f.write(f"{amh}\t{omo}\n")
            
            logger.info(f"💾 Saved {split_name} split to {output_file}")
    
    def get_texts_for_tokenizer(self, language: str = 'both') -> Dict[str, List[str]]:
        """
        Get texts for tokenizer training
        
        Args:
            language: 'amharic', 'oromiffa', or 'both'
            
        Returns:
            Dictionary with language texts
        """
        if not self.amharic_texts:
            self.parse_data()
        
        if language == 'amharic':
            return {'amharic': self.amharic_texts}
        elif language == 'oromiffa':
            return {'oromiffa': self.oromiffa_texts}
        elif language == 'both':
            return {
                'amharic': self.amharic_texts,
                'oromiffa': self.oromiffa_texts
            }
        else:
            raise ValueError(f"Invalid language: {language}. Use 'amharic', 'oromiffa', or 'both'")
