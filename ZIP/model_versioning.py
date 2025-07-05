# Author: KleaSCM
# Date: 2024
# Description: Model versioning module for Kasmeer civil engineering system

import os
import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from utils.logging_utils import setup_logging, log_performance

logger = setup_logging(__name__)

class ModelVersioning:
    # Model versioning and management system
    
    @log_performance(logger)
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.versions_file = self.model_dir / "model_versions.json"
        self.current_version_file = self.model_dir / "current_version.txt"
        
        # Ensure model directory exists
        self.model_dir.mkdir(exist_ok=True)
        logger.info(f"Initialized ModelVersioning with model_dir={model_dir}")
        
    @log_performance(logger)
    def create_version(self, model_files: List[str], metadata: Optional[Dict] = None) -> str:
        # Create a new model version
        logger.info(f"Creating new model version with {len(model_files)} files")
        try:
            # Generate version ID
            version_id = self._generate_version_id()
            logger.debug(f"Generated version ID: {version_id}")
            
            # Create version directory
            version_dir = self.model_dir / f"version_{version_id}"
            version_dir.mkdir(exist_ok=True)
            logger.debug(f"Created version directory: {version_dir}")
            
            # Copy model files to version directory
            copied_files = []
            for file_path in model_files:
                src_path = Path(file_path)
                if src_path.exists():
                    dst_path = version_dir / src_path.name
                    shutil.copy2(src_path, dst_path)
                    copied_files.append(str(dst_path))
                    logger.debug(f"Copied file: {src_path} -> {dst_path}")
                else:
                    logger.warning(f"Model file not found: {file_path}")
            
            logger.info(f"Copied {len(copied_files)} files to version directory")
            
            # Create version metadata
            version_metadata = {
                'version_id': version_id,
                'created_at': datetime.now().isoformat(),
                'files': copied_files,
                'file_hashes': self._calculate_file_hashes(copied_files),
                'metadata': metadata or {}
            }
            
            # Save version metadata
            metadata_file = version_dir / "version_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(version_metadata, f, indent=2)
            logger.debug(f"Saved version metadata to {metadata_file}")
            
            # Update versions registry
            self._update_versions_registry(version_id, version_metadata)
            
            logger.info(f"Successfully created model version: {version_id}")
            return version_id
            
        except Exception as e:
            logger.error(f"Error creating model version: {e}")
            raise
    
    @log_performance(logger)
    def _generate_version_id(self) -> str:
        # Generate a unique version ID based on timestamp and random component
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        import random
        random_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))
        version_id = f"{timestamp}_{random_suffix}"
        logger.debug(f"Generated version ID: {version_id}")
        return version_id
    
    @log_performance(logger)
    def _calculate_file_hashes(self, file_paths: List[str]) -> Dict[str, str]:
        # Calculate SHA256 hashes for model files
        logger.debug(f"Calculating hashes for {len(file_paths)} files")
        hashes = {}
        
        for file_path in file_paths:
            try:
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                    hashes[Path(file_path).name] = file_hash
                    logger.debug(f"Calculated hash for {Path(file_path).name}: {file_hash[:8]}...")
            except Exception as e:
                logger.error(f"Error calculating hash for {file_path}: {e}")
        
        logger.info(f"Calculated hashes for {len(hashes)} files")
        return hashes
    
    @log_performance(logger)
    def _update_versions_registry(self, version_id: str, metadata: Dict):
        # Update the versions registry file
        logger.debug(f"Updating versions registry with version {version_id}")
        try:
            versions = {}
            
            if self.versions_file.exists():
                with open(self.versions_file, 'r') as f:
                    versions = json.load(f)
                logger.debug(f"Loaded existing registry with {len(versions)} versions")
            
            # Add new version
            versions[version_id] = metadata
            
            # Save updated registry
            with open(self.versions_file, 'w') as f:
                json.dump(versions, f, indent=2)
            
            logger.info(f"Updated versions registry: {len(versions)} total versions")
                
        except Exception as e:
            logger.error(f"Error updating versions registry: {e}")
    
    @log_performance(logger)
    def list_versions(self) -> Dict[str, Dict]:
        # List all available model versions
        logger.debug("Listing all model versions")
        versions = {}
        
        try:
            if self.versions_file.exists():
                with open(self.versions_file, 'r') as f:
                    versions = json.load(f)
                logger.debug(f"Found {len(versions)} versions in registry")
            else:
                logger.debug("No versions registry file found")
            
            # Sort by creation date (newest first)
            sorted_versions = dict(sorted(
                versions.items(),
                key=lambda x: x[1]['created_at'],
                reverse=True
            ))
            
            logger.info(f"Listed {len(sorted_versions)} model versions")
            
        except Exception as e:
            logger.error(f"Error listing versions: {e}")
        
        return sorted_versions
    
    @log_performance(logger)
    def get_current_version(self) -> Optional[str]:
        # Get the currently active version
        logger.debug("Getting current version")
        try:
            if self.current_version_file.exists():
                with open(self.current_version_file, 'r') as f:
                    current_version = f.read().strip()
                logger.info(f"Current version: {current_version}")
                return current_version
            else:
                logger.debug("No current version file found")
        except Exception as e:
            logger.error(f"Error reading current version: {e}")
        
        return None
    
    @log_performance(logger)
    def set_current_version(self, version_id: str) -> bool:
        # Set the current active version
        logger.info(f"Setting current version to: {version_id}")
        try:
            # Verify version exists
            versions = self.list_versions()
            if version_id not in versions:
                logger.error(f"Version {version_id} not found in registry")
                return False
            
            # Set as current version
            with open(self.current_version_file, 'w') as f:
                f.write(version_id)
            
            logger.info(f"Successfully set current version to: {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting current version: {e}")
            return False
    
    @log_performance(logger)
    def load_version(self, version_id: str) -> Dict:
        # Load a specific model version
        logger.info(f"Loading model version: {version_id}")
        result = {
            'success': False,
            'version_id': version_id,
            'files': [],
            'metadata': {}
        }
        
        try:
            # Check if version exists
            versions = self.list_versions()
            if version_id not in versions:
                result['error'] = f"Version {version_id} not found"
                logger.error(f"Version {version_id} not found in registry")
                return result
            
            version_metadata = versions[version_id]
            version_dir = self.model_dir / f"version_{version_id}"
            logger.debug(f"Loading version from directory: {version_dir}")
            
            # Verify files exist
            missing_files = []
            for file_path in version_metadata['files']:
                if not Path(file_path).exists():
                    missing_files.append(file_path)
                    logger.warning(f"Missing file: {file_path}")
            
            if missing_files:
                result['error'] = f"Missing files: {missing_files}"
                logger.error(f"Version {version_id} has missing files: {missing_files}")
                return result
            
            # Verify file hashes
            current_hashes = self._calculate_file_hashes(version_metadata['files'])
            stored_hashes = version_metadata['file_hashes']
            
            hash_mismatches = []
            for filename, stored_hash in stored_hashes.items():
                if filename in current_hashes and current_hashes[filename] != stored_hash:
                    hash_mismatches.append(filename)
                    logger.warning(f"Hash mismatch for file: {filename}")
            
            if hash_mismatches:
                result['warning'] = f"Hash mismatches detected: {hash_mismatches}"
                logger.warning(f"Hash mismatches detected for version {version_id}: {hash_mismatches}")
            
            result.update({
                'success': True,
                'files': version_metadata['files'],
                'metadata': version_metadata['metadata'],
                'created_at': version_metadata['created_at']
            })
            
            logger.info(f"Successfully loaded version {version_id} with {len(version_metadata['files'])} files")
            
        except Exception as e:
            logger.error(f"Error loading version {version_id}: {e}")
            result['error'] = str(e)
        
        return result
    
    @log_performance(logger)
    def delete_version(self, version_id: str) -> bool:
        # Delete a model version
        logger.info(f"Deleting model version: {version_id}")
        try:
            # Check if version exists
            versions = self.list_versions()
            if version_id not in versions:
                logger.error(f"Version {version_id} not found in registry")
                return False
            
            # Check if it's the current version
            current_version = self.get_current_version()
            if current_version == version_id:
                logger.error(f"Cannot delete current version: {version_id}")
                return False
            
            # Delete version directory
            version_dir = self.model_dir / f"version_{version_id}"
            if version_dir.exists():
                shutil.rmtree(version_dir)
                logger.debug(f"Deleted version directory: {version_dir}")
            
            # Remove from registry
            del versions[version_id]
            with open(self.versions_file, 'w') as f:
                json.dump(versions, f, indent=2)
            
            logger.info(f"Successfully deleted version: {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting version {version_id}: {e}")
            return False
    
    @log_performance(logger)
    def compare_versions(self, version1: str, version2: str) -> Dict:
        # Compare two model versions
        logger.info(f"Comparing versions: {version1} vs {version2}")
        result = {
            'version1': version1,
            'version2': version2,
            'comparison': {}
        }
        
        try:
            versions = self.list_versions()
            
            if version1 not in versions or version2 not in versions:
                result['error'] = "One or both versions not found"
                logger.error(f"One or both versions not found: {version1}, {version2}")
                return result
            
            v1_meta = versions[version1]
            v2_meta = versions[version2]
            
            # Compare creation dates
            v1_date = datetime.fromisoformat(v1_meta['created_at'])
            v2_date = datetime.fromisoformat(v2_meta['created_at'])
            
            result['comparison'] = {
                'creation_dates': {
                    'version1': v1_meta['created_at'],
                    'version2': v2_meta['created_at'],
                    'difference_days': (v2_date - v1_date).days
                },
                'files': {
                    'version1_files': len(v1_meta['files']),
                    'version2_files': len(v2_meta['files']),
                    'common_files': len(set(v1_meta['files']) & set(v2_meta['files']))
                },
                'metadata': {
                    'version1': v1_meta['metadata'],
                    'version2': v2_meta['metadata']
                }
            }
            
            logger.info(f"Version comparison completed: {result['comparison']['creation_dates']['difference_days']} days difference")
            
        except Exception as e:
            logger.error(f"Error comparing versions: {e}")
            result['error'] = str(e)
        
        return result
    
    @log_performance(logger)
    def get_version_summary(self) -> Dict:
        # Get summary of all versions
        logger.debug("Getting version summary")
        try:
            versions = self.list_versions()
            current_version = self.get_current_version()
            
            summary = {
                'total_versions': len(versions),
                'current_version': current_version,
                'latest_version': list(versions.keys())[0] if versions else None,
                'versions': {}
            }
            
            for version_id, metadata in versions.items():
                summary['versions'][version_id] = {
                    'created_at': metadata['created_at'],
                    'file_count': len(metadata['files']),
                    'is_current': version_id == current_version
                }
            
            logger.info(f"Version summary: {summary['total_versions']} versions, current: {current_version}")
            return summary
            
        except Exception as e:
            logger.error(f"Error getting version summary: {e}")
            return {'error': str(e)} 