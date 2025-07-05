# Author: KleaSCM
# Date: 2024
# Description: Incremental training module for Kasmeer civil engineering system

import os
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from ..utils.logging_utils import setup_logging, log_performance

logger = setup_logging(__name__)

class IncrementalTrainer:
    # Incremental training for neural network models
    
    @log_performance(logger)
    def __init__(self, model_dir: str = "models", data_dir: str = "DataSets"):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.training_history_file = self.model_dir / "training_history.json"
        self.last_training_file = self.model_dir / "last_training_state.pkl"
        logger.info(f"Initialized IncrementalTrainer with model_dir={model_dir}, data_dir={data_dir}")
        
    @log_performance(logger)
    def check_incremental_training_needed(self) -> Dict:
        # Check if incremental training is needed based on data changes
        logger.info("Checking if incremental training is needed")
        result = {
            'needed': False,
            'reason': '',
            'new_data_percentage': 0.0,
            'last_training': None,
            'data_changes': {}
        }
        
        try:
            # Load training history
            if self.training_history_file.exists():
                with open(self.training_history_file, 'r') as f:
                    history = json.load(f)
                result['last_training'] = history.get('last_training_date')
                logger.debug(f"Last training date: {result['last_training']}")
            
            # Check for new data files
            data_changes = self._detect_data_changes()
            result['data_changes'] = data_changes
            
            # Determine if incremental training is needed
            if data_changes['new_files'] or data_changes['modified_files']:
                result['needed'] = True
                result['reason'] = f"Found {len(data_changes['new_files'])} new files and {len(data_changes['modified_files'])} modified files"
                
                # Calculate percentage of new data
                total_files = len(data_changes['all_files'])
                new_files = len(data_changes['new_files']) + len(data_changes['modified_files'])
                if total_files > 0:
                    result['new_data_percentage'] = (new_files / total_files) * 100
                
                logger.info(f"Incremental training needed: {result['reason']}, {result['new_data_percentage']:.1f}% new data")
            else:
                logger.info("No incremental training needed - no data changes detected")
            
        except Exception as e:
            logger.error(f"Error checking incremental training: {e}")
            result['reason'] = f"Error: {e}"
        
        return result
    
    @log_performance(logger)
    def _detect_data_changes(self) -> Dict:
        # Detect changes in data files
        logger.debug("Detecting data file changes")
        changes = {
            'new_files': [],
            'modified_files': [],
            'all_files': [],
            'last_processed': None
        }
        
        try:
            # Load last processing state
            if self.last_training_file.exists():
                with open(self.last_training_file, 'rb') as f:
                    last_state = pickle.load(f)
                changes['last_processed'] = last_state.get('timestamp')
                last_files = last_state.get('files', {})
                logger.debug(f"Last processed timestamp: {changes['last_processed']}")
            else:
                last_files = {}
                logger.debug("No previous training state found")
            
            # Get current files
            current_files = {}
            for file_path in self.data_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix in ['.csv', '.xlsx', '.tif']:
                    current_files[str(file_path)] = {
                        'size': file_path.stat().st_size,
                        'modified': file_path.stat().st_mtime
                    }
            
            changes['all_files'] = list(current_files.keys())
            logger.debug(f"Found {len(current_files)} data files")
            
            # Check for new and modified files
            for file_path, file_info in current_files.items():
                if file_path not in last_files:
                    changes['new_files'].append(file_path)
                    logger.debug(f"New file detected: {file_path}")
                elif file_info['modified'] > last_files[file_path]['modified']:
                    changes['modified_files'].append(file_path)
                    logger.debug(f"Modified file detected: {file_path}")
            
            logger.info(f"Data changes: {len(changes['new_files'])} new, {len(changes['modified_files'])} modified files")
            
        except Exception as e:
            logger.error(f"Error detecting data changes: {e}")
        
        return changes
    
    @log_performance(logger)
    def prepare_incremental_data(self, new_data_only: bool = True) -> Dict:
        # Prepare data for incremental training
        logger.info(f"Preparing incremental data (new_data_only={new_data_only})")
        result = {
            'success': False,
            'training_data': None,
            'validation_data': None,
            'new_samples': 0,
            'total_samples': 0
        }
        
        try:
            # Get data changes
            changes = self._detect_data_changes()
            
            if new_data_only and not (changes['new_files'] or changes['modified_files']):
                result['success'] = True
                result['new_samples'] = 0
                logger.info("No new data to prepare")
                return result
            
            # Load existing data processor
            from src.data.data_processor import DataProcessor
            data_processor = DataProcessor(str(self.data_dir))
            logger.debug("Data processor initialized")
            
            # Load all data (including new)
            data_processor.discover_and_load_all_data()
            logger.debug("All data loaded")
            
            # Prepare training data
            X, y = data_processor.prepare_training_data()
            
            if X is not None and y is not None:
                result['training_data'] = {'X': X, 'y': y}
                result['total_samples'] = len(X)
                result['success'] = True
                
                # Calculate new samples (simplified - could be more sophisticated)
                if changes['new_files'] or changes['modified_files']:
                    result['new_samples'] = max(1, result['total_samples'] // 10)  # Estimate 10% new
                
                logger.info(f"Prepared {result['total_samples']} total samples, {result['new_samples']} new samples")
            else:
                logger.warning("Failed to prepare training data - X or y is None")
            
        except Exception as e:
            logger.error(f"Error preparing incremental data: {e}")
        
        return result
    
    @log_performance(logger)
    def perform_incremental_training(self, model, learning_rate: float = 0.001, 
                                   epochs: int = 10, batch_size: int = 32) -> Dict:
        # Perform incremental training on existing model
        logger.info(f"Starting incremental training: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
        result = {
            'success': False,
            'epochs_trained': 0,
            'final_loss': None,
            'improvement': None,
            'training_time': None
        }
        
        try:
            start_time = datetime.now()
            
            # Prepare incremental data
            data_prep = self.prepare_incremental_data()
            
            if not data_prep['success']:
                result['reason'] = "Failed to prepare incremental data"
                logger.error("Failed to prepare incremental data")
                return result
            
            if data_prep['new_samples'] == 0:
                result['success'] = True
                result['reason'] = "No new data to train on"
                logger.info("No new data to train on")
                return result
            
            # Get training data
            training_data = data_prep['training_data']
            X_train = training_data['X']
            y_train = training_data['y']
            
            # Split for validation
            split_idx = int(0.8 * len(X_train))
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]
            
            logger.info(f"Training split: {len(X_train)} train, {len(X_val)} validation samples")
            
            # Store initial loss
            initial_loss = model.evaluate(X_val, y_val, verbose=0)
            logger.info(f"Initial validation loss: {initial_loss}")
            
            # Perform incremental training
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            
            # Calculate results
            final_loss = model.evaluate(X_val, y_val, verbose=0)
            improvement = initial_loss - final_loss
            
            result.update({
                'success': True,
                'epochs_trained': epochs,
                'final_loss': float(final_loss),
                'improvement': float(improvement),
                'training_time': (datetime.now() - start_time).total_seconds()
            })
            
            logger.info(f"Incremental training completed: final_loss={final_loss:.4f}, improvement={improvement:.4f}, time={result['training_time']:.2f}s")
            
            # Save training state
            self._save_training_state()
            
        except Exception as e:
            logger.error(f"Error in incremental training: {e}")
            result['reason'] = str(e)
        
        return result
    
    @log_performance(logger)
    def _save_training_state(self):
        # Save current training state for future incremental training
        logger.debug("Saving training state")
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'files': {}
            }
            
            # Record current file states
            for file_path in self.data_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix in ['.csv', '.xlsx', '.tif']:
                    state['files'][str(file_path)] = {
                        'size': file_path.stat().st_size,
                        'modified': file_path.stat().st_mtime
                    }
            
            # Save state
            with open(self.last_training_file, 'wb') as f:
                pickle.dump(state, f)
            
            # Update training history
            self._update_training_history()
            
            logger.info(f"Training state saved with {len(state['files'])} files tracked")
            
        except Exception as e:
            logger.error(f"Error saving training state: {e}")
    
    @log_performance(logger)
    def _update_training_history(self):
        # Update training history file
        logger.debug("Updating training history")
        try:
            history = {}
            
            if self.training_history_file.exists():
                with open(self.training_history_file, 'r') as f:
                    history = json.load(f)
            
            # Update history
            history['last_training_date'] = datetime.now().isoformat()
            history['total_training_sessions'] = history.get('total_training_sessions', 0) + 1
            
            # Save updated history
            with open(self.training_history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            logger.info(f"Training history updated: {history['total_training_sessions']} total sessions")
                
        except Exception as e:
            logger.error(f"Error updating training history: {e}")
    
    @log_performance(logger)
    def get_training_history(self) -> Dict:
        # Get training history information
        logger.debug("Retrieving training history")
        history = {
            'last_training': None,
            'total_sessions': 0,
            'training_sessions': []
        }
        
        try:
            if self.training_history_file.exists():
                with open(self.training_history_file, 'r') as f:
                    data = json.load(f)
                
                history.update({
                    'last_training': data.get('last_training_date'),
                    'total_sessions': data.get('total_training_sessions', 0)
                })
                
                logger.info(f"Training history: {history['total_sessions']} sessions, last: {history['last_training']}")
            else:
                logger.debug("No training history file found")
                
        except Exception as e:
            logger.error(f"Error reading training history: {e}")
        
        return history 