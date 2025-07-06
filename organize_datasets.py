#!/usr/bin/env python3
"""
Dataset Organizer - Analyzes and organizes large dataset collections
"""

import os
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import shutil

console = Console()

def analyze_dataset_content(file_path: Path) -> Dict[str, Any]:
    """Analyze a single dataset file to determine its content type"""
    try:
        # Try to read the first few rows to analyze content
        df = pd.read_csv(file_path, nrows=1000)
        
        analysis = {
            'file_name': file_path.name,
            'file_size_mb': file_path.stat().st_size / (1024 * 1024),
            'total_rows': len(pd.read_csv(file_path, nrows=None)) if file_path.stat().st_size < 100 * 1024 * 1024 else 'Large',
            'columns': list(df.columns),
            'content_type': 'unknown',
            'confidence': 0.0,
            'tags': []
        }
        
        # Analyze column names and content to determine type
        column_names = ' '.join(df.columns).lower()
        sample_data = ' '.join(str(df.iloc[:10].values.flatten())).lower()
        
        # Content type detection patterns
        patterns = {
            'construction': ['construction', 'building', 'permit', 'inspection', 'project', 'contractor'],
            'traffic': ['traffic', 'vehicle', 'crash', 'accident', 'road', 'highway', 'transportation'],
            'environmental': ['environmental', 'weather', 'climate', 'air', 'water', 'soil', 'emission'],
            'infrastructure': ['infrastructure', 'pipe', 'drainage', 'storm', 'sewer', 'water main'],
            'energy': ['energy', 'electric', 'solar', 'power', 'consumption', 'utility'],
            'financial': ['financial', 'cost', 'budget', 'revenue', 'expense', 'funding'],
            'geographic': ['geographic', 'boundary', 'parcel', 'lot', 'coordinate', 'latitude'],
            'demographic': ['demographic', 'population', 'census', 'household', 'income'],
            'safety': ['safety', 'violation', 'incident', 'emergency', 'defibrillator'],
            'transportation': ['transportation', 'mta', 'bus', 'subway', 'transit', 'service']
        }
        
        scores = {}
        for content_type, keywords in patterns.items():
            score = 0
            for keyword in keywords:
                if keyword in column_names:
                    score += 2
                if keyword in sample_data:
                    score += 1
            scores[content_type] = score
        
        # Find best match
        if scores:
            best_type = max(scores, key=lambda x: scores[x])
            best_score = scores[best_type]
            analysis['content_type'] = best_type
            analysis['confidence'] = min(best_score / 10.0, 1.0)
            analysis['tags'] = [best_type]
            
            # Add secondary tags
            for content_type, score in scores.items():
                if score > 0 and content_type != best_type:
                    analysis['tags'].append(content_type)
        
        return analysis
        
    except Exception as e:
        return {
            'file_name': file_path.name,
            'file_size_mb': file_path.stat().st_size / (1024 * 1024),
            'error': str(e),
            'content_type': 'error',
            'confidence': 0.0,
            'tags': []
        }

def organize_datasets(data_dir: str, output_dir: Optional[str] = None):
    """Organize datasets by content type"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        console.print(f"[red]❌ Data directory {data_dir} not found[/red]")
        return
    
    # Find all CSV files
    csv_files = list(data_path.glob("*.csv"))
    console.print(f"[blue]📊 Found {len(csv_files)} CSV files to analyze[/blue]")
    
    # Analyze each file
    analyses = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Analyzing datasets...", total=len(csv_files))
        
        for file_path in csv_files:
            progress.update(task, description=f"Analyzing {file_path.name}...")
            analysis = analyze_dataset_content(file_path)
            analyses.append(analysis)
            progress.advance(task)
    
    # Group by content type
    content_groups = {}
    for analysis in analyses:
        content_type = analysis['content_type']
        if content_type not in content_groups:
            content_groups[content_type] = []
        content_groups[content_type].append(analysis)
    
    # Display results
    console.print(f"\n[bold green]🎯 DATASET ORGANIZATION RESULTS[/bold green]")
    console.print("=" * 80)
    
    table = Table(title="Dataset Organization Summary")
    table.add_column("Content Type", style="cyan")
    table.add_column("File Count", style="magenta")
    table.add_column("Total Size (MB)", style="green")
    table.add_column("Average Confidence", style="yellow")
    
    for content_type, files in content_groups.items():
        total_size = sum(f['file_size_mb'] for f in files)
        avg_confidence = sum(f['confidence'] for f in files) / len(files)
        table.add_row(
            content_type.title(),
            str(len(files)),
            f"{total_size:.1f}",
            f"{avg_confidence:.2f}"
        )
    
    console.print(table)
    
    # Show detailed breakdown
    console.print(f"\n[bold]📋 DETAILED BREAKDOWN BY TYPE:[/bold]")
    for content_type, files in content_groups.items():
        console.print(f"\n[cyan]{content_type.upper()}:[/cyan] {len(files)} files")
        for file_info in sorted(files, key=lambda x: x['file_size_mb'], reverse=True)[:5]:
            confidence_color = "green" if file_info['confidence'] > 0.7 else "yellow" if file_info['confidence'] > 0.4 else "red"
            console.print(f"  • {file_info['file_name']} ({file_info['file_size_mb']:.1f}MB, confidence: [{confidence_color}]{file_info['confidence']:.2f}[/{confidence_color}])")
            if file_info['tags']:
                console.print(f"    Tags: {', '.join(file_info['tags'])}")
    
    # Save results
    results = {
        'total_files': len(analyses),
        'content_groups': content_groups,
        'detailed_analyses': analyses
    }
    
    with open('dataset_organization_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    console.print(f"\n[green]✅ Results saved to dataset_organization_results.json[/green]")
    
    # Create organized directory structure
    if output_dir is not None:
        organize_files_by_type(analyses, data_path, output_dir)

def organize_files_by_type(analyses: List[Dict], source_dir: Path, output_dir: str):
    """Create organized directory structure by content type"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    console.print(f"\n[blue]📁 Creating organized directory structure in {output_dir}...[/blue]")
    
    for analysis in analyses:
        if analysis['content_type'] == 'error':
            continue
            
        content_type = analysis['content_type']
        type_dir = output_path / content_type
        type_dir.mkdir(exist_ok=True)
        
        source_file = source_dir / analysis['file_name']
        dest_file = type_dir / analysis['file_name']
        
        try:
            if not dest_file.exists():
                shutil.copy2(source_file, dest_file)
                console.print(f"  ✅ Copied {analysis['file_name']} to {content_type}/")
        except Exception as e:
            console.print(f"  ❌ Failed to copy {analysis['file_name']}: {e}")
    
    console.print(f"\n[green]✅ Dataset organization complete![/green]")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "/home/klea/Documents/Dev/AI/DataSets"
    
    output_dir = None
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    organize_datasets(data_dir, output_dir) 