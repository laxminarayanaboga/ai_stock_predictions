"""
Model Version Management System
Handles creating, saving, and managing different versions of stock prediction models
"""

import os
import json
import shutil
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ModelVersionManager:
    """Manages different versions of trained models"""
    
    def __init__(self, base_models_dir: str = "models"):
        """
        Initialize the version manager
        
        Args:
            base_models_dir: Base directory for all models
        """
        self.base_dir = Path(base_models_dir)
        self.versions_dir = self.base_dir / "versions"
        self.current_dir = self.base_dir
        
        # Create directories
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        
        # Version registry file
        self.registry_file = self.versions_dir / "model_registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load the model registry or create a new one"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "versions": {},
                "current_version": None,
                "creation_date": datetime.now().isoformat()
            }
    
    def _save_registry(self):
        """Save the model registry"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def create_version(self, 
                      version_name: str, 
                      description: str = "", 
                      model_type: str = "EnhancedStockLSTM",
                      copy_current: bool = True) -> str:
        """
        Create a new model version
        
        Args:
            version_name: Name for this version (e.g., "v2_improved_features")
            description: Description of changes in this version
            model_type: Type of model architecture
            copy_current: Whether to copy current model as starting point
            
        Returns:
            Version directory path
        """
        # Create version directory
        version_dir = self.versions_dir / version_name
        version_dir.mkdir(exist_ok=True)
        
        # Copy current model if requested
        if copy_current and (self.current_dir / "enhanced_stock_lstm.pth").exists():
            shutil.copy2(
                self.current_dir / "enhanced_stock_lstm.pth",
                version_dir / "model.pth"
            )
            if (self.current_dir / "model_metadata.json").exists():
                shutil.copy2(
                    self.current_dir / "model_metadata.json",
                    version_dir / "metadata.json"
                )
        
        # Create version info
        version_info = {
            "version_name": version_name,
            "description": description,
            "model_type": model_type,
            "creation_date": datetime.now().isoformat(),
            "training_date": None,
            "performance_metrics": {},
            "model_parameters": {},
            "training_config": {},
            "status": "created"
        }
        
        # Save version info
        with open(version_dir / "version_info.json", 'w') as f:
            json.dump(version_info, f, indent=2)
        
        # Update registry
        self.registry["versions"][version_name] = {
            "path": str(version_dir),
            "creation_date": version_info["creation_date"],
            "description": description,
            "status": "created"
        }
        
        self._save_registry()
        
        print(f"✅ Created model version: {version_name}")
        print(f"📁 Path: {version_dir}")
        
        return str(version_dir)
    
    def list_versions(self) -> List[Dict]:
        """List all available model versions"""
        versions = []
        for name, info in self.registry["versions"].items():
            version_dir = Path(info["path"])
            
            # Load detailed info if available
            version_info_file = version_dir / "version_info.json"
            if version_info_file.exists():
                with open(version_info_file, 'r') as f:
                    detailed_info = json.load(f)
                    versions.append(detailed_info)
            else:
                versions.append({
                    "version_name": name,
                    "description": info.get("description", ""),
                    "creation_date": info.get("creation_date", ""),
                    "status": info.get("status", "unknown")
                })
        
        return sorted(versions, key=lambda x: x.get("creation_date", ""), reverse=True)
    
    def get_version_path(self, version_name: str) -> Optional[Path]:
        """Get the path to a specific version"""
        if version_name in self.registry["versions"]:
            return Path(self.registry["versions"][version_name]["path"])
        return None
    
    def set_current_version(self, version_name: str):
        """Set a version as the current active version"""
        version_path = self.get_version_path(version_name)
        if not version_path or not version_path.exists():
            raise ValueError(f"Version {version_name} not found")
        
        # Copy model files to current directory
        model_file = version_path / "model.pth"
        metadata_file = version_path / "metadata.json"
        
        if model_file.exists():
            shutil.copy2(model_file, self.current_dir / "enhanced_stock_lstm.pth")
        
        if metadata_file.exists():
            shutil.copy2(metadata_file, self.current_dir / "model_metadata.json")
        
        # Update registry
        self.registry["current_version"] = version_name
        self._save_registry()
        
        print(f"✅ Set {version_name} as current version")
    
    def save_trained_model(self, 
                          version_name: str,
                          model: torch.nn.Module,
                          metadata: Dict,
                          training_config: Dict,
                          performance_metrics: Dict):
        """
        Save a trained model to a specific version
        
        Args:
            version_name: Version to save to
            model: Trained PyTorch model
            metadata: Model metadata
            training_config: Training configuration used
            performance_metrics: Performance metrics achieved
        """
        version_path = self.get_version_path(version_name)
        if not version_path:
            raise ValueError(f"Version {version_name} not found")
        
        # Save model
        torch.save(model.state_dict(), version_path / "model.pth")
        
        # Save metadata
        with open(version_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update version info
        version_info_file = version_path / "version_info.json"
        with open(version_info_file, 'r') as f:
            version_info = json.load(f)
        
        version_info.update({
            "training_date": datetime.now().isoformat(),
            "performance_metrics": performance_metrics,
            "model_parameters": {
                "input_size": metadata.get("input_size"),
                "hidden_size": metadata.get("hidden_size"),
                "num_layers": metadata.get("num_layers"),
                "output_size": metadata.get("output_size")
            },
            "training_config": training_config,
            "status": "trained"
        })
        
        with open(version_info_file, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        # Update registry
        self.registry["versions"][version_name]["status"] = "trained"
        self._save_registry()
        
        print(f"✅ Saved trained model to version: {version_name}")
    
    def compare_versions(self, version_names: List[str] = None) -> 'pd.DataFrame':
        """Compare performance metrics across versions"""        
        if version_names is None:
            version_names = list(self.registry["versions"].keys())
        
        comparison_data = []
        
        for version_name in version_names:
            version_path = self.get_version_path(version_name)
            if not version_path:
                continue
            
            version_info_file = version_path / "version_info.json"
            if version_info_file.exists():
                with open(version_info_file, 'r') as f:
                    info = json.load(f)
                
                metrics = info.get("performance_metrics", {})
                params = info.get("model_parameters", {})
                
                comparison_data.append({
                    "Version": version_name,
                    "Status": info.get("status", "unknown"),
                    "Training Date": info.get("training_date", "Not trained"),
                    "MSE": metrics.get("test_mse", "N/A"),
                    "MAE": metrics.get("test_mae", "N/A"),
                    "RMSE": metrics.get("test_rmse", "N/A"),
                    "Hidden Size": params.get("hidden_size", "N/A"),
                    "Layers": params.get("num_layers", "N/A"),
                    "Description": info.get("description", "")
                })
        
        return pd.DataFrame(comparison_data)
    
    def delete_version(self, version_name: str, confirm: bool = False):
        """Delete a model version"""
        if not confirm:
            print(f"⚠️  This will permanently delete version: {version_name}")
            print("Use confirm=True to proceed")
            return
        
        version_path = self.get_version_path(version_name)
        if version_path and version_path.exists():
            shutil.rmtree(version_path)
            
            # Remove from registry
            if version_name in self.registry["versions"]:
                del self.registry["versions"][version_name]
            
            # Update current version if needed
            if self.registry.get("current_version") == version_name:
                self.registry["current_version"] = None
            
            self._save_registry()
            print(f"✅ Deleted version: {version_name}")
        else:
            print(f"❌ Version {version_name} not found")


def main():
    """Demo of version management system"""
    vm = ModelVersionManager()
    
    print("📦 Model Version Manager Demo")
    print("=" * 40)
    
    # List existing versions
    versions = vm.list_versions()
    if versions:
        print(f"Found {len(versions)} existing versions:")
        for v in versions:
            print(f"  - {v['version_name']}: {v.get('description', 'No description')}")
    else:
        print("No versions found")
    
    # Create a new version
    # vm.create_version(
    #     "v1_baseline", 
    #     "Baseline model with current architecture",
    #     copy_current=True
    # )


if __name__ == "__main__":
    main()