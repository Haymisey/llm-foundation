"""
Tokenizer Trainer for Amharic-Oromiffa Translation

This module trains BPE tokenizers for both languages using
the Hugging Face tokenizers library.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenizerTrainer:
    """
    Trainer for BPE tokenizers for Amharic and Oromiffa
    """
    
    def __init__(self, vocab_size: int = 8000, min_frequency: int = 2):
        """
        Initialize tokenizer trainer
        
        Args:
            vocab_size: Maximum vocabulary size for each tokenizer
            min_frequency: Minimum frequency for tokens to be included
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Special tokens
        self.special_tokens = [
            "<PAD>",      # Padding token
            "<START>",    # Start of sequence
            "<END>",      # End of sequence
            "<UNK>",      # Unknown token
            "<SEP>"       # Separator between languages
        ]
        
        # Tokenizers
        self.amharic_tokenizer = None
        self.oromiffa_tokenizer = None
        
        logger.info(f"Initializing tokenizer trainer with vocab_size={vocab_size}")
    
    def train_amharic_tokenizer(self, texts: List[str], 
                               save_path: str = "tokenizers/amharic_tokenizer.json") -> Tokenizer:
        """
        Train BPE tokenizer for Amharic
        
        Args:
            texts: List of Amharic texts for training
            save_path: Path to save the trained tokenizer
            
        Returns:
            Trained Amharic tokenizer
        """
        logger.info(f"🚀 Training Amharic BPE tokenizer on {len(texts)} texts...")
        
        # Initialize tokenizer
        tokenizer = Tokenizer(BPE())
        
        # Pre-tokenizer (ByteLevel for better handling of Unicode)
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        
        # Decoder
        tokenizer.decoder = decoders.ByteLevel()
        
        # Trainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=True
        )
        
        # Train tokenizer
        tokenizer.train_from_iterator(texts, trainer=trainer)
        
        # Post-processing: add special tokens
        tokenizer.post_processor = TemplateProcessing(
            single="<START> $A <END>",
            pair="<START> $A <SEP> $B <END>",
            special_tokens=[
                ("<START>", tokenizer.token_to_id("<START>")),
                ("<END>", tokenizer.token_to_id("<END>")),
                ("<SEP>", tokenizer.token_to_id("<SEP>"))
            ]
        )
        
        # Save tokenizer
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(save_path))
        
        logger.info(f"✅ Amharic tokenizer trained and saved to {save_path}")
        logger.info(f"   Vocabulary size: {tokenizer.get_vocab_size()}")
        
        self.amharic_tokenizer = tokenizer
        return tokenizer
    
    def train_oromiffa_tokenizer(self, texts: List[str], 
                                save_path: str = "tokenizers/oromiffa_tokenizer.json") -> Tokenizer:
        """
        Train BPE tokenizer for Oromiffa
        
        Args:
            texts: List of Oromiffa texts for training
            save_path: Path to save the trained tokenizer
            
        Returns:
            Trained Oromiffa tokenizer
        """
        logger.info(f"🚀 Training Oromiffa BPE tokenizer on {len(texts)} texts...")
        
        # Initialize tokenizer
        tokenizer = Tokenizer(BPE())
        
        # Pre-tokenizer (ByteLevel for better handling of Latin script)
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        
        # Decoder
        tokenizer.decoder = decoders.ByteLevel()
        
        # Trainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=True
        )
        
        # Train tokenizer
        tokenizer.train_from_iterator(texts, trainer=trainer)
        
        # Post-processing: add special tokens
        tokenizer.post_processor = TemplateProcessing(
            single="<START> $A <END>",
            pair="<START> $A <SEP> $B <END>",
            special_tokens=[
                ("<START>", tokenizer.token_to_id("<START>")),
                ("<END>", tokenizer.token_to_id("<END>")),
                ("<SEP>", tokenizer.token_to_id("<SEP>"))
            ]
        )
        
        # Save tokenizer
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(save_path))
        
        logger.info(f"✅ Oromiffa tokenizer trained and saved to {save_path}")
        logger.info(f"   Vocabulary size: {tokenizer.get_vocab_size()}")
        
        self.oromiffa_tokenizer = tokenizer
        return tokenizer
    
    def train_both_tokenizers(self, amharic_texts: List[str], 
                             oromiffa_texts: List[str],
                             save_dir: str = "tokenizers") -> Tuple[Tokenizer, Tokenizer]:
        """
        Train both Amharic and Oromiffa tokenizers
        
        Args:
            amharic_texts: List of Amharic texts
            oromiffa_texts: List of Oromiffa texts
            save_dir: Directory to save tokenizers
            
        Returns:
            Tuple of (amharic_tokenizer, oromiffa_tokenizer)
        """
        logger.info("🚀 Training both Amharic and Oromiffa tokenizers...")
        
        # Train Amharic tokenizer
        amharic_path = f"{save_dir}/amharic_tokenizer.json"
        amharic_tokenizer = self.train_amharic_tokenizer(amharic_texts, amharic_path)
        
        # Train Oromiffa tokenizer
        oromiffa_path = f"{save_dir}/oromiffa_tokenizer.json"
        oromiffa_tokenizer = self.train_oromiffa_tokenizer(oromiffa_texts, oromiffa_path)
        
        # Save tokenizer info
        self._save_tokenizer_info(save_dir)
        
        logger.info("✅ Both tokenizers trained successfully!")
        return amharic_tokenizer, oromiffa_tokenizer
    
    def _save_tokenizer_info(self, save_dir: str):
        """Save tokenizer information and statistics"""
        save_path = Path(save_dir)
        info_file = save_path / "tokenizer_info.json"
        
        info = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "special_tokens": self.special_tokens,
            "amharic_tokenizer": {
                "vocab_size": self.amharic_tokenizer.get_vocab_size() if self.amharic_tokenizer else None,
                "file": "amharic_tokenizer.json"
            },
            "oromiffa_tokenizer": {
                "vocab_size": self.oromiffa_tokenizer.get_vocab_size() if self.oromiffa_tokenizer else None,
                "file": "oromiffa_tokenizer.json"
            }
        }
        
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Tokenizer info saved to {info_file}")
    
    def load_tokenizers(self, amharic_path: str = "tokenizers/amharic_tokenizer.json",
                       oromiffa_path: str = "tokenizers/oromiffa_tokenizer.json") -> Tuple[Tokenizer, Tokenizer]:
        """
        Load pre-trained tokenizers
        
        Args:
            amharic_path: Path to Amharic tokenizer
            oromiffa_path: Path to Oromiffa tokenizer
            
        Returns:
            Tuple of (amharic_tokenizer, oromiffa_tokenizer)
        """
        logger.info("📂 Loading pre-trained tokenizers...")
        
        # Load Amharic tokenizer
        if Path(amharic_path).exists():
            self.amharic_tokenizer = Tokenizer.from_file(amharic_path)
            logger.info(f"✅ Loaded Amharic tokenizer from {amharic_path}")
        else:
            raise FileNotFoundError(f"Amharic tokenizer not found: {amharic_path}")
        
        # Load Oromiffa tokenizer
        if Path(oromiffa_path).exists():
            self.oromiffa_tokenizer = Tokenizer.from_file(oromiffa_path)
            logger.info(f"✅ Loaded Oromiffa tokenizer from {oromiffa_path}")
        else:
            raise FileNotFoundError(f"Oromiffa tokenizer not found: {oromiffa_path}")
        
        return self.amharic_tokenizer, self.oromiffa_tokenizer
    
    def test_tokenizers(self, sample_texts: List[str] = None):
        """
        Test the trained tokenizers with sample texts
        
        Args:
            sample_texts: List of sample texts to test (optional)
        """
        if not self.amharic_tokenizer or not self.oromiffa_tokenizer:
            logger.error("Tokenizers not loaded. Train or load them first.")
            return
        
        if sample_texts is None:
            sample_texts = [
                "ሂድ።",  # Amharic: Go
                "Deemi.",  # Oromiffa: Go
                "እሺ።",  # Amharic: Okay
                "Akkam."   # Oromiffa: Okay
            ]
        
        logger.info("🧪 Testing tokenizers with sample texts...")
        
        for text in sample_texts:
            # Determine language (simple heuristic)
            if any('\u1200' <= char <= '\u137F' for char in text):  # Amharic Unicode range
                tokenizer = self.amharic_tokenizer
                lang = "Amharic"
            else:
                tokenizer = self.oromiffa_tokenizer
                lang = "Oromiffa"
            
            # Tokenize
            encoding = tokenizer.encode(text)
            tokens = encoding.tokens
            ids = encoding.ids
            
            logger.info(f"   {lang}: '{text}'")
            logger.info(f"      Tokens: {tokens}")
            logger.info(f"      IDs: {ids}")
            logger.info(f"      Length: {len(tokens)} tokens")
            logger.info("")
    
    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes for both tokenizers"""
        if not self.amharic_tokenizer or not self.oromiffa_tokenizer:
            return {}
        
        return {
            'amharic': self.amharic_tokenizer.get_vocab_size(),
            'oromiffa': self.oromiffa_tokenizer.get_vocab_size()
        }
