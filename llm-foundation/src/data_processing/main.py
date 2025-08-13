"""
Main Data Processing Pipeline for Amharic-Oromiffa Translation

This script demonstrates the complete data processing pipeline:
1. Parse the parallel text dataset
2. Train BPE tokenizers for both languages
3. Create vocabulary management system
4. Build PyTorch DataLoaders for training
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from dataset_parser import AmharicOromiffaDataset
from tokenizer_trainer import TokenizerTrainer
from vocabulary import Vocabulary
from data_loader import TranslationDataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main data processing pipeline"""
    logger.info("🚀 Starting Amharic-Oromiffa Data Processing Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Parse the dataset
    logger.info("\n📊 Step 1: Parsing Dataset")
    logger.info("-" * 30)
    
    dataset = AmharicOromiffaDataset("data/amh_omo.txt")
    amharic_texts, oromiffa_texts = dataset.parse_data()
    
    # Step 2: Split data into train/val/test
    logger.info("\n📊 Step 2: Creating Data Splits")
    logger.info("-" * 30)
    
    data_splits = dataset.get_training_data(
        train_split=0.8,
        val_split=0.1,
        test_split=0.1
    )
    
    # Save splits for later use
    dataset.save_splits("data/processed")
    
    # Step 3: Train BPE tokenizers
    logger.info("\n🚀 Step 3: Training BPE Tokenizers")
    logger.info("-" * 30)
    
    tokenizer_trainer = TokenizerTrainer(vocab_size=8000, min_frequency=2)
    
    # Get texts for tokenizer training
    train_amharic = [text for text, _ in data_splits['train']]
    train_oromiffa = [text for _, text in data_splits['train']]
    
    # Train both tokenizers
    amharic_tokenizer, oromiffa_tokenizer = tokenizer_trainer.train_both_tokenizers(
        train_amharic, 
        train_oromiffa,
        save_dir="tokenizers"
    )
    
    # Test tokenizers
    tokenizer_trainer.test_tokenizers()
    
    # Step 4: Create vocabulary management
    logger.info("\n📚 Step 4: Creating Vocabulary Management")
    logger.info("-" * 30)
    
    vocabulary = Vocabulary(amharic_tokenizer, oromiffa_tokenizer)
    
    # Test vocabulary functionality
    vocabulary.test_vocabulary()
    
    # Save vocabulary info
    vocabulary.save_vocabulary_info("vocabulary_info.json")
    
    # Step 5: Create PyTorch DataLoaders
    logger.info("\n🔧 Step 5: Creating PyTorch DataLoaders")
    logger.info("-" * 30)
    
    data_loader = TranslationDataLoader(
        vocabulary=vocabulary,
        batch_size=32,
        max_length=128,  # Maximum sequence length
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    # Create datasets and DataLoaders
    data_loader.create_datasets(data_splits)
    data_loader.create_data_loaders()
    
    # Test DataLoader
    data_loader.test_data_loader(num_batches=2)
    
    # Save DataLoader info
    data_loader.save_data_loader_info("data_loader_info.json")
    
    # Step 6: Summary and next steps
    logger.info("\n✅ Data Processing Pipeline Complete!")
    logger.info("=" * 60)
    
    # Get final statistics
    vocab_stats = vocabulary.get_vocabulary_stats()
    batch_info = data_loader.get_batch_info()
    
    logger.info("📊 Final Statistics:")
    logger.info(f"   Amharic vocabulary size: {vocab_stats['amharic_vocab_size']}")
    logger.info(f"   Oromiffa vocabulary size: {vocab_stats['oromiffa_vocab_size']}")
    logger.info(f"   Model vocabulary size: {vocab_stats['model_vocab_size']}")
    
    if 'train' in batch_info:
        train_info = batch_info['train']
        logger.info(f"   Training batches: {train_info['num_batches']}")
        logger.info(f"   Batch size: {train_info['batch_size']}")
        logger.info(f"   Input shape: {train_info['amharic_shape']}")
        logger.info(f"   Target shape: {train_info['oromiffa_shape']}")
    
    logger.info("\n🎯 Next Steps:")
    logger.info("   1. Use the trained tokenizers for inference")
    logger.info("   2. Integrate DataLoaders with PyTorch training loop")
    logger.info("   3. Train transformer model on the processed data")
    logger.info("   4. Evaluate translation quality")
    
    logger.info("\n💾 Files Created:")
    logger.info("   - tokenizers/amharic_tokenizer.json")
    logger.info("   - tokenizers/oromiffa_tokenizer.json")
    logger.info("   - tokenizers/tokenizer_info.json")
    logger.info("   - vocabulary_info.json")
    logger.info("   - data_loader_info.json")
    logger.info("   - data/processed/ (train/val/test splits)")
    
    return {
        'dataset': dataset,
        'tokenizer_trainer': tokenizer_trainer,
        'vocabulary': vocabulary,
        'data_loader': data_loader,
        'data_splits': data_splits
    }

def test_pipeline():
    """Test the pipeline with a small subset"""
    logger.info("🧪 Testing Pipeline with Small Dataset")
    logger.info("=" * 50)
    
    # Create a small test dataset
    test_data = [
        ("ሂድ።", "Deemi."),
        ("እሺ።", "Akkam."),
        ("እንደ ምን አለ?", "Eenyu?"),
        ("አዎ!", "Ajaa'iba!"),
        ("ደህና ነው!", "Daakiyyee!")
    ]
    
    # Test tokenizer training on small data
    tokenizer_trainer = TokenizerTrainer(vocab_size=100, min_frequency=1)
    
    amharic_texts = [text for text, _ in test_data]
    oromiffa_texts = [text for _, text in test_data]
    
    # Train tokenizers
    amharic_tokenizer, oromiffa_tokenizer = tokenizer_trainer.train_both_tokenizers(
        amharic_texts, oromiffa_texts, save_dir="tokenizers_test"
    )
    
    # Test vocabulary
    vocabulary = Vocabulary(amharic_tokenizer, oromiffa_tokenizer)
    vocabulary.test_vocabulary(test_data)
    
    logger.info("✅ Test pipeline completed successfully!")

if __name__ == "__main__":
    try:
        # Run the main pipeline
        results = main()
        logger.info("🎉 Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        logger.info("Trying test pipeline...")
        
        try:
            test_pipeline()
        except Exception as e2:
            logger.error(f"❌ Test pipeline also failed: {e2}")
            sys.exit(1)
