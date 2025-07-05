# Author: KleaSCM
# Date: 2024
# Description: Plotting utilities for the Kasmeer civil engineering neural network system

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from utils.logging_utils import setup_logging, log_performance

logger = setup_logging(__name__)

class DataVisualizer:
    # Data visualization utilities for civil engineering data
    
    @log_performance(logger)
    def __init__(self, output_dir: str = "visuals"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logger
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        logger.info(f"Initialized DataVisualizer with output_dir={output_dir}")
    
    @log_performance(logger)
    def plot_data_summary(self, data_processor, save: bool = True):
        # Create summary plots of the data
        logger.info("Creating data summary plots")
        try:
            summary = data_processor.get_data_summary()
            logger.debug(f"Retrieved data summary with {len(summary)} data types")
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Civil Engineering Data Summary', fontsize=16, fontweight='bold')
            
            # Plot 1: Data types and record counts
            data_types = list(summary.keys())
            record_counts = [summary[dt].get('rows', 0) for dt in data_types if isinstance(summary[dt], dict)]
            
            axes[0, 0].bar(data_types[:len(record_counts)], record_counts)
            axes[0, 0].set_title('Record Counts by Data Type')
            axes[0, 0].set_ylabel('Number of Records')
            axes[0, 0].tick_params(axis='x', rotation=45)
            logger.debug(f"Created record counts plot with {len(record_counts)} data types")
            
            # Plot 2: Memory usage
            memory_usage = [summary[dt].get('memory_usage', 0) / (1024*1024) for dt in data_types if isinstance(summary[dt], dict)]
            
            axes[0, 1].pie(memory_usage, labels=data_types[:len(memory_usage)], autopct='%1.1f%%')
            axes[0, 1].set_title('Memory Usage by Data Type')
            logger.debug(f"Created memory usage pie chart")
            
            # Plot 3: Column counts
            column_counts = [len(summary[dt].get('columns', [])) for dt in data_types if isinstance(summary[dt], dict)]
            
            axes[1, 0].bar(data_types[:len(column_counts)], column_counts)
            axes[1, 0].set_title('Column Counts by Data Type')
            axes[1, 0].set_ylabel('Number of Columns')
            axes[1, 0].tick_params(axis='x', rotation=45)
            logger.debug(f"Created column counts plot")
            
            # Plot 4: Data completeness
            completeness = [summary[dt].get('completeness', 0) for dt in data_types if isinstance(summary[dt], dict)]
            
            axes[1, 1].bar(data_types[:len(completeness)], completeness)
            axes[1, 1].set_title('Data Completeness by Type')
            axes[1, 1].set_ylabel('Completeness (%)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            logger.debug(f"Created data completeness plot")
            
            plt.tight_layout()
            
            if save:
                output_path = self.output_dir / "data_summary.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Data summary plot saved to {output_path}")
            
            plt.show()
            logger.info("Data summary plot completed successfully")
            
        except Exception as e:
            logger.error(f"Error creating data summary plot: {e}")
    
    @log_performance(logger)
    def plot_risk_assessment(self, risk_data: dict, location: tuple, save: bool = True):
        # Plot risk assessment results
        logger.info(f"Creating risk assessment plot for location {location}")
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'Risk Assessment at {location}', fontsize=14, fontweight='bold')
            
            # Risk scores
            risk_types = list(risk_data.keys())
            risk_scores = list(risk_data.values())
            logger.debug(f"Plotting {len(risk_types)} risk types: {risk_types}")
            
            # Bar plot
            bars = axes[0].bar(risk_types, risk_scores, color=['red', 'orange', 'yellow'])
            axes[0].set_title('Risk Scores by Category')
            axes[0].set_ylabel('Risk Score (0-1)')
            axes[0].set_ylim(0, 1)
            axes[0].tick_params(axis='x', rotation=45)
            
            # Color bars based on risk level
            for bar, score in zip(bars, risk_scores):
                if score > 0.7:
                    bar.set_color('red')
                elif score > 0.4:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')
            
            # Pie chart
            axes[1].pie(risk_scores, labels=risk_types, autopct='%1.1f%%')
            axes[1].set_title('Risk Distribution')
            
            plt.tight_layout()
            
            if save:
                output_path = self.output_dir / f"risk_assessment_{location[0]}_{location[1]}.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Risk assessment plot saved to {output_path}")
            
            plt.show()
            logger.info("Risk assessment plot completed successfully")
            
        except Exception as e:
            logger.error(f"Error creating risk assessment plot: {e}")
    
    @log_performance(logger)
    def plot_training_history(self, history: dict, save: bool = True):
        """Plot training history"""
        logger.info("Creating training history plot")
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Training History', fontsize=14, fontweight='bold')
            
            # Loss plot
            if 'train_loss' in history and 'val_loss' in history:
                epochs = range(1, len(history['train_loss']) + 1)
                axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
                axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
                axes[0].set_title('Model Loss')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].legend()
                axes[0].grid(True)
                logger.debug(f"Created loss plot with {len(epochs)} epochs")
            
            # Metrics plot
            if 'metrics' in history:
                metrics = history['metrics']
                metric_names = list(metrics.keys())
                metric_values = list(metrics.values())
                logger.debug(f"Plotting {len(metric_names)} metrics: {metric_names}")
                
                bars = axes[1].bar(metric_names, metric_values)
                axes[1].set_title('Final Metrics')
                axes[1].set_ylabel('Score')
                axes[1].tick_params(axis='x', rotation=45)
                
                # Color bars based on performance
                for bar, value in zip(bars, metric_values):
                    if value > 0.8:
                        bar.set_color('green')
                    elif value > 0.6:
                        bar.set_color('orange')
                    else:
                        bar.set_color('red')
            
            plt.tight_layout()
            
            if save:
                output_path = self.output_dir / "training_history.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Training history plot saved to {output_path}")
            
            plt.show()
            logger.info("Training history plot completed successfully")
            
        except Exception as e:
            logger.error(f"Error creating training history plot: {e}")
    
    @log_performance(logger)
    def plot_feature_importance(self, feature_names: list, importance_scores: list, save: bool = True):
        """Plot feature importance"""
        logger.info(f"Creating feature importance plot for {len(feature_names)} features")
        try:
            # Sort features by importance
            feature_importance = list(zip(feature_names, importance_scores))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Take top 10 features
            top_features = feature_importance[:10]
            names, scores = zip(*top_features)
            logger.debug(f"Plotting top 10 features: {names[:3]}...")
            
            plt.figure(figsize=(10, 6))
            bars = plt.barh(range(len(names)), scores)
            plt.yticks(range(len(names)), names)
            plt.xlabel('Importance Score')
            plt.title('Top 10 Feature Importance')
            plt.gca().invert_yaxis()
            
            # Color bars
            for i, bar in enumerate(bars):
                if i < 3:
                    bar.set_color('red')
                elif i < 6:
                    bar.set_color('orange')
                else:
                    bar.set_color('blue')
            
            plt.tight_layout()
            
            if save:
                output_path = self.output_dir / "feature_importance.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature importance plot saved to {output_path}")
            
            plt.show()
            logger.info("Feature importance plot completed successfully")
            
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {e}")
    
    @log_performance(logger)
    def plot_infrastructure_map(self, data_processor, save: bool = True):
        """Plot infrastructure data on a map"""
        # TODO: Implement infrastructure mapping
        # TODO: Add interactive map functionality
        # TODO: Include different infrastructure types
        logger.info("Creating infrastructure map")
        
        try:
            if 'infrastructure' not in data_processor.processed_data:
                logger.warning("No infrastructure data available for mapping")
                return
            
            infra_df = data_processor.processed_data['infrastructure']
            logger.debug(f"Infrastructure data: {len(infra_df)} records")
            
            if infra_df.empty or 'estimated_lat' not in infra_df.columns:
                logger.warning("Infrastructure data missing coordinates")
                return
            
            # Create scatter plot of infrastructure
            plt.figure(figsize=(12, 8))
            plt.scatter(infra_df['estimated_lon'], infra_df['estimated_lat'], 
                       alpha=0.6, s=20, c='blue')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title('Infrastructure Distribution')
            plt.grid(True, alpha=0.3)
            
            if save:
                output_path = self.output_dir / "infrastructure_map.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Infrastructure map saved to {output_path}")
            
            plt.show()
            logger.info("Infrastructure map completed successfully")
            
        except Exception as e:
            logger.error(f"Error creating infrastructure map: {e}")
    
    @log_performance(logger)
    def plot_climate_data(self, data_processor, save: bool = True):
        """Plot climate data visualization"""
        # TODO: Implement climate data visualization
        # TODO: Add seasonal climate patterns
        # TODO: Include climate change analysis
        logger.info("Creating climate data visualization")
        
        try:
            if 'climate' not in data_processor.processed_data:
                logger.warning("No climate data available for visualization")
                return
            
            climate_data = data_processor.processed_data['climate']
            logger.debug(f"Climate data variables: {list(climate_data.keys())}")
            
            if not climate_data:
                logger.warning("Climate data is empty")
                return
            
            # Create subplots for different climate variables
            n_vars = len(climate_data)
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Climate Data Overview', fontsize=16, fontweight='bold')
            
            # Flatten axes for easier iteration
            axes_flat = axes.flatten()
            
            for i, (var_name, var_data) in enumerate(climate_data.items()):
                if i >= len(axes_flat):
                    break
                
                if 'data' in var_data:
                    data = var_data['data']
                    im = axes_flat[i].imshow(data, cmap='viridis')
                    axes_flat[i].set_title(f'{var_name.replace("_", " ").title()}')
                    plt.colorbar(im, ax=axes_flat[i])
                    logger.debug(f"Plotted climate variable: {var_name}")
            
            plt.tight_layout()
            
            if save:
                output_path = self.output_dir / "climate_data.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Climate data plot saved to {output_path}")
            
            plt.show()
            logger.info("Climate data visualization completed successfully")
            
        except Exception as e:
            logger.error(f"Error creating climate data plot: {e}")
    
    @log_performance(logger)
    def create_dashboard(self, data_processor, neural_network, save: bool = True):
        """Create a comprehensive dashboard"""
        # TODO: Implement comprehensive dashboard
        # TODO: Add interactive widgets
        # TODO: Include real-time data updates
        logger.info("Creating comprehensive dashboard")
        
        try:
            fig = plt.figure(figsize=(20, 12))
            fig.suptitle('Kasmeer Civil Engineering Dashboard', fontsize=20, fontweight='bold')
            
            # Create grid layout
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            
            # Data summary
            ax1 = fig.add_subplot(gs[0, :2])
            summary = data_processor.get_data_summary()
            data_types = list(summary.keys())
            record_counts = [summary[dt].get('rows', 0) for dt in data_types if isinstance(summary[dt], dict)]
            ax1.bar(data_types[:len(record_counts)], record_counts)
            ax1.set_title('Data Summary')
            ax1.set_ylabel('Record Count')
            ax1.tick_params(axis='x', rotation=45)
            logger.debug(f"Added data summary to dashboard: {len(record_counts)} data types")
            
            # Model status
            ax2 = fig.add_subplot(gs[0, 2:])
            model_summary = neural_network.get_model_summary()
            status_text = f"Model Loaded: {model_summary.get('model_loaded', False)}\n"
            status_text += f"Device: {model_summary.get('device', 'Unknown')}\n"
            status_text += f"Features: {model_summary.get('feature_count', 0)}\n"
            status_text += f"Outputs: {model_summary.get('output_count', 0)}"
            ax2.text(0.1, 0.5, status_text, transform=ax2.transAxes, fontsize=12, 
                    verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax2.set_title('Model Status')
            ax2.axis('off')
            logger.debug("Added model status to dashboard")
            
            # Infrastructure map
            ax3 = fig.add_subplot(gs[1:, :2])
            if 'infrastructure' in data_processor.processed_data:
                infra_df = data_processor.processed_data['infrastructure']
                if not infra_df.empty and 'estimated_lat' in infra_df.columns:
                    ax3.scatter(infra_df['estimated_lon'], infra_df['estimated_lat'], 
                               alpha=0.6, s=10, c='blue')
                    logger.debug(f"Added infrastructure map with {len(infra_df)} points")
            ax3.set_title('Infrastructure Distribution')
            ax3.set_xlabel('Longitude')
            ax3.set_ylabel('Latitude')
            ax3.grid(True, alpha=0.3)
            
            # Risk assessment example
            ax4 = fig.add_subplot(gs[1, 2:])
            # TODO: Add actual risk assessment visualization
            ax4.text(0.1, 0.5, "Risk Assessment\n(Example Location)", 
                    transform=ax4.transAxes, fontsize=12, 
                    verticalalignment='center', horizontalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            ax4.set_title('Risk Assessment')
            ax4.axis('off')
            logger.debug("Added risk assessment placeholder to dashboard")
            
            # System info
            ax5 = fig.add_subplot(gs[2, 2:])
            # TODO: Add system information display
            ax5.text(0.1, 0.5, "System Information\n• Data loaded\n• Model status\n• Performance metrics", 
                    transform=ax5.transAxes, fontsize=10, 
                    verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            ax5.set_title('System Info')
            ax5.axis('off')
            logger.debug("Added system info to dashboard")
            
            if save:
                output_path = self.output_dir / "dashboard.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Dashboard saved to {output_path}")
            
            plt.show()
            logger.info("Dashboard created successfully")
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}") 