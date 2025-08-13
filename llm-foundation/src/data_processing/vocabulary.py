"""
Vocabulary Management for Amharic-Oromiffa Translation

This module handles vocabulary management, special tokens,
and provides utilities for working with tokenized sequences.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tokenizers import Tokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Vocabulary:
    """
    Vocabulary management for translation model
    """
    
    def __init__(self, amharic_tokenizer: Tokenizer, oromiffa_tokenizer: Tokenizer):
        """
        Initialize vocabulary manager
        
        Args:
            amharic_tokenizer: Trained Amharic tokenizer
            oromiffa_tokenizer: Trained Oromiffa tokenizer
        """
        self.amharic_tokenizer = amharic_tokenizer
        self.oromiffa_tokenizer = oromiffa_tokenizer
        
        # Special token IDs
        self.special_tokens = {
            'PAD': 0,
            'START': 1, 
            'END': 2,
            'UNK': 3,
            'SEP': 4
        }
        
        # Get vocabulary sizes
        self.amharic_vocab_size = amharic_tokenizer.get_vocab_size()
        self.oromiffa_vocab_size = oromiffa_tokenizer.get_vocab_size()
        
        # Use the larger vocabulary size for the model
        self.model_vocab_size = max(self.amharic_vocab_size, self.oromiffa_vocab_size)
        
        logger.info(f"📚 Vocabulary initialized:")
        logger.info(f"   Amharic vocab size: {self.amharic_vocab_size}")
        logger.info(f"   Oromiffa vocab size: {self.oromiffa_vocab_size}")
        logger.info(f"   Model vocab size: {self.model_vocab_size}")
    
    def get_special_token_ids(self) -> Dict[str, int]:
        """Get special token IDs"""
        return self.special_tokens.copy()
    
    def get_pad_token_id(self) -> int:
        """Get padding token ID"""
        return self.special_tokens['PAD']
    
    def get_start_token_id(self) -> int:
        """Get start token ID"""
        return self.special_tokens['START']
    
    def get_end_token_id(self) -> int:
        """Get end token ID"""
        return self.special_tokens['END']
    
    def get_unk_token_id(self) -> int:
        """Get unknown token ID"""
        return self.special_tokens['UNK']
    
    def get_sep_token_id(self) -> int:
        """Get separator token ID"""
        return self.special_tokens['SEP']
    
    def encode_amharic(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode Amharic text to token IDs
        
        Args:
            text: Amharic text to encode
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token IDs
        """
        encoding = self.amharic_tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids
    
    def encode_oromiffa(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode Oromiffa text to token IDs
        
        Args:
            text: Oromiffa text to encode
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token IDs
        """
        encoding = self.oromiffa_tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids
    
    def decode_amharic(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode Amharic token IDs back to text
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded Amharic text
        """
        return self.amharic_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def decode_oromiffa(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode Oromiffa token IDs back to text
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded Oromiffa text
        """
        return self.oromiffa_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def encode_pair(self, amharic_text: str, oromiffa_text: str, 
                   max_length: Optional[int] = None) -> Tuple[List[int], List[int]]:
        """
        Encode a pair of Amharic-Oromiffa texts
        
        Args:
            amharic_text: Source Amharic text
            oromiffa_text: Target Oromiffa text
            max_length: Maximum sequence length (optional)
            
        Returns:
            Tuple of (amharic_ids, oromiffa_ids)
        """
        # Encode both texts
        amharic_ids = self.encode_amharic(amharic_text, add_special_tokens=True)
        oromiffa_ids = self.encode_oromiffa(oromiffa_text, add_special_tokens=True)
        
        # Truncate if max_length specified
        if max_length:
            amharic_ids = amharic_ids[:max_length]
            oromiffa_ids = oromiffa_ids[:max_length]
        
        return amharic_ids, oromiffa_ids
    
    def pad_sequences(self, sequences: List[List[int]], 
                     max_length: Optional[int] = None,
                     padding: str = 'post',
                     truncating: str = 'post') -> List[List[int]]:
        """
        Pad sequences to the same length
        
        Args:
            sequences: List of sequences to pad
            max_length: Maximum length (if None, use max sequence length)
            padding: 'pre' or 'post' padding
            truncating: 'pre' or 'post' truncating
            
        Returns:
            List of padded sequences
        """
        if not sequences:
            return []
        
        # Determine max length
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        padded_sequences = []
        pad_token_id = self.get_pad_token_id()
        
        for seq in sequences:
            # Truncate if necessary
            if len(seq) > max_length:
                if truncating == 'post':
                    seq = seq[:max_length]
                else:  # pre
                    seq = seq[-max_length:]
            
            # Pad if necessary
            if len(seq) < max_length:
                padding_length = max_length - len(seq)
                if padding == 'post':
                    seq = seq + [pad_token_id] * padding_length
                else:  # pre
                    seq = [pad_token_id] * padding_length + seq
            
            padded_sequences.append(seq)
        
        return padded_sequences
    
    def create_attention_mask(self, sequences: List[List[int]], 
                             pad_token_id: Optional[int] = None) -> List[List[int]]:
        """
        Create attention mask for padded sequences
        
        Args:
            sequences: List of padded sequences
            pad_token_id: Padding token ID (if None, use default)
            
        Returns:
            List of attention masks (1 for real tokens, 0 for padding)
        """
        if pad_token_id is None:
            pad_token_id = self.get_pad_token_id()
        
        attention_masks = []
        
        for seq in sequences:
            mask = [1 if token_id != pad_token_id else 0 for token_id in seq]
            attention_masks.append(mask)
        
        return attention_masks
    
    def get_vocabulary_stats(self) -> Dict[str, Union[int, Dict[str, int]]]:
        """
        Get comprehensive vocabulary statistics
        
        Returns:
            Dictionary with vocabulary statistics
        """
        stats = {
            'amharic_vocab_size': self.amharic_vocab_size,
            'oromiffa_vocab_size': self.oromiffa_vocab_size,
            'model_vocab_size': self.model_vocab_size,
            'special_tokens': self.special_tokens.copy(),
            'special_token_count': len(self.special_tokens)
        }
        
        return stats
    
    def save_vocabulary_info(self, save_path: str = "vocabulary_info.json"):
        """
        Save vocabulary information to file
        
        Args:
            save_path: Path to save vocabulary info
        """
        info = {
            'vocabulary_stats': self.get_vocabulary_stats(),
            'special_tokens': self.special_tokens,
            'amharic_tokenizer_file': 'amharic_tokenizer.json',
            'oromiffa_tokenizer_file': 'oromiffa_tokenizer.json'
        }
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Vocabulary info saved to {save_path}")
    
    def test_vocabulary(self, sample_texts: List[Tuple[str, str]] = None):
        """
        Test vocabulary functionality with sample texts
        
        Args:
            sample_texts: List of (amharic, oromiffa) text pairs
        """
        if sample_texts is None:
            sample_texts = [
                ("ሂድ።", "Deemi."),
                ("እሺ።", "Akkam."),
                ("እንደ ምን አለ?", "Eenyu?")
            ]
        
        logger.info("🧪 Testing vocabulary functionality...")
        
        for amh_text, omo_text in sample_texts:
            logger.info(f"\n   Testing: '{amh_text}' → '{omo_text}'")
            
            # Encode
            amh_ids, omo_ids = self.encode_pair(amh_text, omo_text)
            logger.info(f"      Amharic IDs: {amh_ids}")
            logger.info(f"      Oromiffa IDs: {omo_ids}")
            
            # Decode
            amh_decoded = self.decode_amharic(amh_ids)
            omo_decoded = self.decode_oromiffa(omo_ids)
            logger.info(f"      Amharic decoded: '{amh_decoded}'")
            logger.info(f"      Oromiffa decoded: '{omo_decoded}'")
            
            # Pad sequences
            padded_amh, padded_omo = self.pad_sequences([amh_ids, omo_ids], max_length=10)
            logger.info(f"      Padded Amharic: {padded_amh}")
            logger.info(f"      Padded Oromiffa: {padded_omo}")
            
            # Attention mask
            attention_mask = self.create_attention_mask([amh_ids, omo_ids])
            logger.info(f"      Attention mask: {attention_mask}")
