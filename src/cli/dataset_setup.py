# Author: KleaSCM
# Date: 2024
# Description: Interactive dataset setup CLI for companies

import click
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.text import Text
from ..utils.logging_utils import setup_logging, log_performance
from src.core.dataset_config import DatasetConfig

console = Console()
logger = setup_logging(__name__)

@click.group()
def dataset_setup():
    """Dataset setup and configuration commands"""
    pass

@dataset_setup.command()
@click.option('--config-path', default='config.yaml', help='Path to configuration file')
@log_performance(logger)
def discover(config_path):
    """Discover datasets in your data directory"""
    logger.info(f"Discovering datasets with config: {config_path}")
    
    console.print(Panel.fit(
        "[bold blue]🔍 Dataset Discovery[/bold blue]\n"
        "This will scan your data directory and identify available datasets.",
        title="Kasmeer Dataset Setup"
    ))
    
    try:
        dataset_config = DatasetConfig(config_path)
        discovered_datasets = dataset_config.discover_datasets()
        
        if not discovered_datasets:
            console.print("[red]❌ No datasets found![/red]")
            console.print("\n[bold]Possible reasons:[/bold]")
            console.print("• Data directory is empty")
            console.print("• File patterns don't match your files")
            console.print("• Files are in unsupported formats")
            console.print(f"\n[bold]Data directory:[/bold] {dataset_config.data_dir}")
            return
        
        # Display discovered datasets
        table = Table(title="Discovered Datasets")
        table.add_column("Dataset Type", style="cyan")
        table.add_column("Files Found", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("File Types", style="yellow")
        
        for dataset_type, info in discovered_datasets.items():
            files = info.get('files', [])
            file_types = list(set(f.get('file_type', 'unknown') for f in files))
            status = "✅ Ready" if info.get('enabled', True) else "❌ Disabled"
            
            table.add_row(
                dataset_type.replace('_', ' ').title(),
                str(len(files)),
                status,
                ', '.join(file_types)
            )
        
        console.print(table)
        
        # Show file details
        console.print("\n[bold]File Details:[/bold]")
        for dataset_type, info in discovered_datasets.items():
            files = info.get('files', [])
            if files:
                console.print(f"\n[cyan]{dataset_type.replace('_', ' ').title()}:[/cyan]")
                for file_info in files[:3]:  # Show first 3 files
                    console.print(f"  • {file_info['name']} ({file_info['file_type']})")
                if len(files) > 3:
                    console.print(f"  • ... and {len(files) - 3} more files")
        
        console.print(f"\n[green]✅ Discovery complete! Found {len(discovered_datasets)} dataset types.[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Error during discovery: {e}[/red]")
        logger.error(f"Discovery error: {e}")

@dataset_setup.command()
@click.option('--config-path', default='config.yaml', help='Path to configuration file')
@click.option('--dataset-type', help='Specific dataset type to configure')
@log_performance(logger)
def configure(config_path, dataset_type):
    """Configure dataset settings interactively"""
    logger.info(f"Configuring datasets with config: {config_path}")
    
    console.print(Panel.fit(
        "[bold blue]⚙️ Dataset Configuration[/bold blue]\n"
        "This will help you configure your datasets for optimal processing.",
        title="Kasmeer Dataset Setup"
    ))
    
    try:
        dataset_config = DatasetConfig(config_path)
        discovered_datasets = dataset_config.discover_datasets()
        
        if not discovered_datasets:
            console.print("[red]❌ No datasets found! Run 'discover' first.[/red]")
            return
        
        # Select dataset to configure
        if dataset_type:
            if dataset_type not in discovered_datasets:
                console.print(f"[red]❌ Dataset type '{dataset_type}' not found![/red]")
                return
            selected_datasets = [dataset_type]
        else:
            # Let user choose
            console.print("\n[bold]Available datasets:[/bold]")
            for i, (dt, info) in enumerate(discovered_datasets.items(), 1):
                status = "✅" if info.get('enabled', True) else "❌"
                console.print(f"{i}. {status} {dt.replace('_', ' ').title()}")
            
            choice = Prompt.ask(
                "\nSelect dataset to configure",
                choices=[str(i) for i in range(1, len(discovered_datasets) + 1)]
            )
            selected_datasets = [list(discovered_datasets.keys())[int(choice) - 1]]
        
        # Configure each selected dataset
        for dt in selected_datasets:
            console.print(f"\n[bold cyan]Configuring {dt.replace('_', ' ').title()}:[/bold cyan]")
            _configure_dataset(dataset_config, dt, discovered_datasets[dt])
        
        # Save configuration
        if Confirm.ask("\nSave configuration changes?"):
            if dataset_config.save_configuration():
                console.print("[green]✅ Configuration saved successfully![/green]")
            else:
                console.print("[red]❌ Failed to save configuration![/red]")
        
    except Exception as e:
        console.print(f"[red]❌ Error during configuration: {e}[/red]")
        logger.error(f"Configuration error: {e}")

def _configure_dataset(dataset_config: DatasetConfig, dataset_type: str, dataset_info: Dict):
    """Configure a specific dataset"""
    config = dataset_info.get('config', {})
    files = dataset_info.get('files', [])
    
    if not files:
        console.print(f"[yellow]⚠️ No files found for {dataset_type}[/yellow]")
        return
    
    # Show current configuration
    console.print(f"\n[bold]Current Configuration:[/bold]")
    console.print(f"• Enabled: {config.get('enabled', True)}")
    console.print(f"• Priority: {config.get('priority', 999)}")
    console.print(f"• File patterns: {', '.join(config.get('file_patterns', []))}")
    console.print(f"• Required columns: {', '.join(config.get('required_columns', []))}")
    
    # Configure basic settings
    enabled = Confirm.ask(f"\nEnable {dataset_type} dataset?", default=config.get('enabled', True))
    priority = int(Prompt.ask("Processing priority (1-10, lower = higher priority)", 
                             default=str(config.get('priority', 999))))
    
    # Configure file patterns
    console.print(f"\n[bold]File Patterns:[/bold]")
    current_patterns = config.get('file_patterns', [])
    for i, pattern in enumerate(current_patterns):
        console.print(f"{i+1}. {pattern}")
    
    if Confirm.ask("Add new file patterns?"):
        new_pattern = Prompt.ask("Enter file pattern (e.g., '*my_data*')")
        if new_pattern:
            current_patterns.append(new_pattern)
    
    # Configure column mappings
    if files and files[0].get('file_type') == 'tabular':
        _configure_column_mappings(dataset_config, dataset_type, files[0])
    
    # Update configuration
    config.update({
        'enabled': enabled,
        'priority': priority,
        'file_patterns': current_patterns
    })

def _configure_column_mappings(dataset_config: DatasetConfig, dataset_type: str, file_info: Dict):
    """Configure column mappings for tabular data"""
    console.print(f"\n[bold]Column Mappings for {dataset_type}:[/bold]")
    
    columns = file_info.get('columns', [])
    console.print(f"Available columns: {', '.join(columns)}")
    
    # Show current mappings
    current_mappings = dataset_config.company_config.get('data_mappings', {}).get('column_mappings', {})
    
    if current_mappings:
        console.print(f"\n[bold]Current mappings:[/bold]")
        for original, mapped in current_mappings.items():
            console.print(f"• {original} → {mapped}")
    
    if Confirm.ask("Configure column mappings?"):
        while True:
            original_col = Prompt.ask("Enter original column name (or 'done' to finish)")
            if original_col.lower() == 'done':
                break
            
            if original_col in columns:
                mapped_col = Prompt.ask(f"Map '{original_col}' to standard name")
                if mapped_col:
                    current_mappings[original_col] = mapped_col
                    console.print(f"[green]✅ Mapped {original_col} → {mapped_col}[/green]")
            else:
                console.print(f"[red]❌ Column '{original_col}' not found![/red]")

@dataset_setup.command()
@click.option('--config-path', default='config.yaml', help='Path to configuration file')
@click.option('--output-path', help='Output path for the template')
@log_performance(logger)
def create_template(config_path, output_path):
    """Create a template configuration for a new dataset type"""
    logger.info(f"Creating dataset template with config: {config_path}")
    
    console.print(Panel.fit(
        "[bold blue]📝 Dataset Template Creation[/bold blue]\n"
        "This will help you create a template for a new dataset type.",
        title="Kasmeer Dataset Setup"
    ))
    
    try:
        dataset_config = DatasetConfig(config_path)
        
        # Get dataset type name
        dataset_type = Prompt.ask("Enter dataset type name (e.g., 'soil_data', 'traffic_data')")
        if not dataset_type:
            console.print("[red]❌ Dataset type name is required![/red]")
            return
        
        # Create template
        template = dataset_config.create_dataset_template(dataset_type)
        
        # Customize template
        console.print(f"\n[bold]Template for '{dataset_type}':[/bold]")
        
        # File patterns
        patterns = Prompt.ask("Enter file patterns (comma-separated)", 
                             default="*" + dataset_type.lower() + "*")
        template['file_patterns'] = [p.strip() for p in patterns.split(',')]
        
        # Required columns
        required_cols = Prompt.ask("Enter required columns (comma-separated)")
        if required_cols:
            template['required_columns'] = [col.strip() for col in required_cols.split(',')]
        
        # Optional columns
        optional_cols = Prompt.ask("Enter optional columns (comma-separated)")
        if optional_cols:
            template['optional_columns'] = [col.strip() for col in optional_cols.split(',')]
        
        # Coordinate generation
        if Confirm.ask("Enable coordinate generation?"):
            template['coordinate_generation']['enabled'] = True
            template['coordinate_generation']['method'] = Prompt.ask(
                "Coordinate generation method", 
                choices=['random_distribution', 'clustering'],
                default='random_distribution'
            )
        
        # Save template
        if output_path:
            template_path = Path(output_path)
        else:
            template_path = Path(f"template_{dataset_type}.yaml")
        
        with open(template_path, 'w') as f:
            yaml.dump({dataset_type: template}, f, default_flow_style=False, indent=2)
        
        console.print(f"[green]✅ Template saved to {template_path}[/green]")
        console.print(f"\n[bold]Next steps:[/bold]")
        console.print(f"1. Add this template to your config.yaml")
        console.print(f"2. Place your data files in the DataSets directory")
        console.print(f"3. Run 'discover' to find your new dataset")
        
    except Exception as e:
        console.print(f"[red]❌ Error creating template: {e}[/red]")
        logger.error(f"Template creation error: {e}")

@dataset_setup.command()
@click.option('--config-path', default='config.yaml', help='Path to configuration file')
@log_performance(logger)
def validate(config_path):
    """Validate dataset configuration"""
    logger.info(f"Validating dataset configuration: {config_path}")
    
    console.print(Panel.fit(
        "[bold blue]✅ Configuration Validation[/bold blue]\n"
        "This will validate your dataset configuration and check for issues.",
        title="Kasmeer Dataset Setup"
    ))
    
    try:
        dataset_config = DatasetConfig(config_path)
        
        # Validate configuration
        validation_results = dataset_config.validate_dataset_configuration()
        
        # Display results
        if validation_results['valid']:
            console.print("[green]✅ Configuration is valid![/green]")
        else:
            console.print("[red]❌ Configuration has errors![/red]")
        
        # Show errors
        if validation_results['errors']:
            console.print(f"\n[bold red]Errors ({len(validation_results['errors'])}):[/bold red]")
            for error in validation_results['errors']:
                console.print(f"• {error}")
        
        # Show warnings
        if validation_results['warnings']:
            console.print(f"\n[bold yellow]Warnings ({len(validation_results['warnings'])}):[/bold yellow]")
            for warning in validation_results['warnings']:
                console.print(f"• {warning}")
        
        # Show dataset status
        console.print(f"\n[bold]Dataset Status:[/bold]")
        table = Table()
        table.add_column("Dataset Type", style="cyan")
        table.add_column("Valid", style="green")
        table.add_column("Errors", style="red")
        table.add_column("Warnings", style="yellow")
        
        for dataset_type, result in validation_results['datasets'].items():
            table.add_row(
                dataset_type.replace('_', ' ').title(),
                "✅" if result['valid'] else "❌",
                str(len(result['errors'])),
                str(len(result['warnings']))
            )
        
        console.print(table)
        
        # Company information
        company_info = dataset_config.get_company_info()
        console.print(f"\n[bold]Company Information:[/bold]")
        console.print(f"• Name: {company_info['name']}")
        console.print(f"• Region: {company_info['region']}")
        
    except Exception as e:
        console.print(f"[red]❌ Error during validation: {e}[/red]")
        logger.error(f"Validation error: {e}")

@dataset_setup.command()
@click.option('--config-path', default='config.yaml', help='Path to configuration file')
@log_performance(logger)
def company_info(config_path):
    """Set company information"""
    logger.info(f"Setting company information: {config_path}")
    
    console.print(Panel.fit(
        "[bold blue]🏢 Company Information[/bold blue]\n"
        "This will help you configure your company-specific settings.",
        title="Kasmeer Dataset Setup"
    ))
    
    try:
        dataset_config = DatasetConfig(config_path)
        company_info = dataset_config.get_company_info()
        
        console.print(f"\n[bold]Current Company Information:[/bold]")
        console.print(f"• Name: {company_info['name']}")
        console.print(f"• Region: {company_info['region']}")
        
        if Confirm.ask("\nUpdate company information?"):
            # Get new company information
            new_name = Prompt.ask("Company name", default=company_info['name'])
            new_region = Prompt.ask("Region", default=company_info['region'])
            
            # Update configuration
            dataset_config.company_config['name'] = new_name
            dataset_config.company_config['region'] = new_region
            
            # Save configuration
            if dataset_config.save_configuration():
                console.print("[green]✅ Company information updated successfully![/green]")
            else:
                console.print("[red]❌ Failed to update company information![/red]")
        
    except Exception as e:
        console.print(f"[red]❌ Error setting company information: {e}[/red]")
        logger.error(f"Company info error: {e}")

if __name__ == '__main__':
    dataset_setup() 