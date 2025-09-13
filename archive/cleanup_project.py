#!/usr/bin/env python3
"""
Project Cleanup Script
Moves model-specific scripts from root to appropriate model directories
and cleans up temporary files while preserving important project structure
"""

import os
import shutil
from pathlib import Path


def cleanup_root_folder():
    """Clean up root folder by moving/organizing files"""
    
    root_dir = Path("/Users/bogalaxminarayana/myGit/ai_stock_predictions")
    
    # Files to move to V4 model directory (if exists)
    v4_files = [
        "generate_v4_predictions.py"
    ]
    
    # Files to move to utilities
    utility_files = [
        "convert_10min_cache_to_csv.py"
    ]
    
    # Generic prediction files to move to archive
    archive_files = [
        "predict.py"
    ]
    
    # Log files to clean (but keep as they might be needed)
    log_files = [
        "fyersApi.log",
        "fyersRequests.log"
    ]
    
    print("🧹 Starting project cleanup...")
    
    # Create V4 model directory if it doesn't exist
    v4_dir = root_dir / "models" / "versions" / "v4_minimal"
    if not v4_dir.exists():
        v4_dir.mkdir(parents=True)
        print(f"✅ Created V4 model directory: {v4_dir}")
    
    # Create scripts directory for V4
    v4_scripts_dir = v4_dir / "scripts"
    if not v4_scripts_dir.exists():
        v4_scripts_dir.mkdir()
        print(f"✅ Created V4 scripts directory: {v4_scripts_dir}")
    
    # Create utilities/scripts directory if it doesn't exist
    utils_scripts_dir = root_dir / "utilities" / "scripts"
    if not utils_scripts_dir.exists():
        utils_scripts_dir.mkdir(parents=True)
        print(f"✅ Created utilities scripts directory: {utils_scripts_dir}")
    
    # Create archive directory
    archive_dir = root_dir / "archive"
    if not archive_dir.exists():
        archive_dir.mkdir()
        print(f"✅ Created archive directory: {archive_dir}")
    
    # Move V4-specific files
    for file in v4_files:
        src = root_dir / file
        dst = v4_scripts_dir / file
        if src.exists():
            shutil.move(str(src), str(dst))
            print(f"📁 Moved {file} to V4 scripts directory")
    
    # Move utility files
    for file in utility_files:
        src = root_dir / file
        dst = utils_scripts_dir / file
        if src.exists():
            shutil.move(str(src), str(dst))
            print(f"🔧 Moved {file} to utilities directory")
    
    # Move archive files
    for file in archive_files:
        src = root_dir / file
        dst = archive_dir / file
        if src.exists():
            shutil.move(str(src), str(dst))
            print(f"📦 Moved {file} to archive directory")
    
    # Clean up __pycache__ in root
    pycache_dir = root_dir / "__pycache__"
    if pycache_dir.exists():
        shutil.rmtree(pycache_dir)
        print("🗑️  Removed __pycache__ from root")
    
    # Clean up __init__.py in root (not needed)
    init_file = root_dir / "__init__.py"
    if init_file.exists():
        init_file.unlink()
        print("🗑️  Removed __init__.py from root")
    
    # Move log files to a logs directory
    logs_dir = root_dir / "logs"
    if not logs_dir.exists():
        logs_dir.mkdir()
        print(f"✅ Created logs directory: {logs_dir}")
    
    for file in log_files:
        src = root_dir / file
        dst = logs_dir / file
        if src.exists():
            shutil.move(str(src), str(dst))
            print(f"📊 Moved {file} to logs directory")
    
    print("\n🎉 Cleanup completed!")
    print_clean_structure()


def print_clean_structure():
    """Print the clean project structure"""
    print("\n📂 Clean Project Structure:")
    print("ai_stock_predictions/")
    print("├── README.md                 # Main documentation")
    print("├── PROJECT_ORGANIZATION.md   # Project organization guide")
    print("├── requirements.txt          # Dependencies")
    print("├── config.ini               # Configuration")
    print("├── data/                    # Data files")
    print("├── models/                  # All models")
    print("│   ├── versions/            # Organized by version")
    print("│   │   ├── v2_attention/    # V2 (best so far)")
    print("│   │   ├── v4_minimal/      # V4 with moved scripts")
    print("│   │   └── v5_enhanced/     # V5 (latest)")
    print("├── src/                     # Source modules")
    print("├── api/                     # API integration")
    print("├── utilities/               # Helper utilities")
    print("│   └── scripts/            # Utility scripts")
    print("├── logs/                    # Log files")
    print("├── archive/                 # Old/deprecated files")
    print("└── simulation_results/      # Trading results")


def create_v4_documentation():
    """Create basic documentation for V4 model"""
    v4_dir = Path("/Users/bogalaxminarayana/myGit/ai_stock_predictions/models/versions/v4_minimal")
    
    readme_content = """# Model V4 Minimal

**Creation Date**: September 2025  
**Status**: Completed  
**Focus**: Minimal feature set for faster training

## Overview

V4 was designed as a minimal version to test faster training with reduced features.

## Scripts

- `generate_v4_predictions.py` - Generate predictions using V4 model

## Performance

TBD - Performance metrics to be documented after evaluation.

## Notes

This version focused on minimal complexity for faster experimentation.
"""
    
    readme_path = v4_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"📝 Created V4 documentation: {readme_path}")


def main():
    """Main cleanup function"""
    print("🚀 AI Stock Predictions - Project Cleanup")
    print("=" * 50)
    
    try:
        cleanup_root_folder()
        create_v4_documentation()
        
        print("\n✨ Project cleanup completed successfully!")
        print("\n🎯 Next Steps:")
        print("1. Train V5 model: cd models/versions/v5_enhanced/scripts && python train_v5.py")
        print("2. Make predictions: python predict_next_day.py")
        print("3. Run backtesting: python historical_simulation.py")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")


if __name__ == "__main__":
    main()