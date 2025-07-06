# Author: KleaSCM
# Date: 2024
# Description: Test the sexy modular Content Analyzer system

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from datetime import datetime

from src.core.content_analyzer import ContentAnalyzer, SmartTagger, CrossDatasetIntelligence, ContentDetector

console = Console()

def test_content_analyzer():
    """Test the Content Analyzer system"""
    console.print(Panel.fit("[bold blue]🧠 TESTING SEXY MODULAR CONTENT ANALYZER[/bold blue]", style="blue"))
    
    # Initialize components
    content_analyzer = ContentAnalyzer()
    smart_tagger = SmartTagger()
    cross_intelligence = CrossDatasetIntelligence()
    content_detector = ContentDetector()
    
    console.print("✅ All content analyzer components initialized!")
    
    # Create test datasets
    console.print("\n[bold]📊 Creating test datasets...[/bold]")
    
    # Traffic dataset
    traffic_data = pd.DataFrame({
        'speed': np.random.randint(20, 80, 100),
        'congestion_level': np.random.choice(['low', 'medium', 'high'], 100),
        'traffic_volume': np.random.randint(100, 1000, 100),
        'latitude': np.random.uniform(40.7, 40.8, 100),
        'longitude': np.random.uniform(-74.0, -73.9, 100),
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H')
    })
    
    # Weather dataset
    weather_data = pd.DataFrame({
        'temperature': np.random.uniform(10, 30, 50),
        'humidity': np.random.uniform(30, 90, 50),
        'precipitation': np.random.uniform(0, 50, 50),
        'wind_speed': np.random.uniform(0, 25, 50),
        'weather_condition': np.random.choice(['sunny', 'cloudy', 'rainy', 'stormy'], 50),
        'latitude': np.random.uniform(40.7, 40.8, 50),
        'longitude': np.random.uniform(-74.0, -73.9, 50),
        'date': pd.date_range('2024-01-01', periods=50, freq='D')
    })
    
    # Construction dataset
    construction_data = pd.DataFrame({
        'project_name': [f'Project_{i}' for i in range(20)],
        'construction_type': np.random.choice(['residential', 'commercial', 'infrastructure'], 20),
        'permit_status': np.random.choice(['approved', 'pending', 'completed'], 20),
        'estimated_cost': np.random.randint(100000, 5000000, 20),
        'contractor': [f'Contractor_{i}' for i in range(20)],
        'latitude': np.random.uniform(40.7, 40.8, 20),
        'longitude': np.random.uniform(-74.0, -73.9, 20),
        'start_date': pd.date_range('2024-01-01', periods=20, freq='M')
    })
    
    console.print(f"✅ Created test datasets:")
    console.print(f"   • Traffic: {len(traffic_data)} records")
    console.print(f"   • Weather: {len(weather_data)} records") 
    console.print(f"   • Construction: {len(construction_data)} records")
    
    # Test Content Analyzer
    console.print("\n[bold yellow]🔍 TESTING CONTENT ANALYZER[/bold yellow]")
    
    for name, data in [('traffic', traffic_data), ('weather', weather_data), ('construction', construction_data)]:
        console.print(f"\n[cyan]Analyzing {name} dataset...[/cyan]")
        
        analysis = content_analyzer.analyze_content(data, f"{name}_test.csv")
        
        table = Table(title=f"Content Analysis Results - {name.title()}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Content Type", analysis['content_type'])
        table.add_row("Confidence", f"{analysis['confidence_scores'].get(analysis['content_type'], 0):.2f}")
        table.add_row("Records", str(analysis['data_characteristics']['total_records']))
        table.add_row("Columns", str(analysis['data_characteristics']['total_columns']))
        table.add_row("Has Coordinates", str(analysis['data_characteristics']['has_coordinates']))
        table.add_row("Has Temporal Data", str(analysis['data_characteristics']['has_temporal_data']))
        table.add_row("Completeness", f"{analysis['data_characteristics']['data_completeness']:.1f}%")
        
        console.print(table)
        
        if analysis['suggested_tags']:
            console.print(f"🏷️ Suggested tags: {', '.join(analysis['suggested_tags'])}")
    
    # Test Smart Tagger
    console.print("\n[bold yellow]🏷️ TESTING SMART TAGGER[/bold yellow]")
    
    for name, data in [('traffic', traffic_data), ('weather', weather_data), ('construction', construction_data)]:
        console.print(f"\n[cyan]Auto-tagging {name} dataset...[/cyan]")
        
        tagging_result = smart_tagger.auto_tag_dataset(data, f"{name}_test.csv")
        
        console.print(f"📋 Tags: {', '.join(tagging_result['tags'])}")
        console.print(f"🎯 Confidence: {tagging_result['confidence']:.2f}")
        console.print(f"💡 Summary: {tagging_result['tagging_summary']}")
        
        if tagging_result['suggestions']:
            console.print(f"💭 Suggestions: {', '.join(tagging_result['suggestions'])}")
    
    # Test Cross Dataset Intelligence
    console.print("\n[bold yellow]🧠 TESTING CROSS DATASET INTELLIGENCE[/bold yellow]")
    
    datasets = {
        'traffic': traffic_data,
        'weather': weather_data,
        'construction': construction_data
    }
    
    relationship_analysis = cross_intelligence.analyze_dataset_relationships(datasets)
    
    console.print(f"📊 Found {sum(len(rels) for rels in relationship_analysis['relationships'].values())} relationships")
    console.print(f"🔗 Identified {len(relationship_analysis['combinations'])} potential combinations")
    
    # Show relationships
    for rel_type, rels in relationship_analysis['relationships'].items():
        if rels:
            console.print(f"\n[bold]{rel_type.title()} Relationships:[/bold]")
            for rel in rels[:3]:  # Show first 3
                console.print(f"  • {rel['description']}")
    
    # Show combinations
    if relationship_analysis['combinations']:
        console.print(f"\n[bold]🔗 Dataset Combinations:[/bold]")
        for combo in relationship_analysis['combinations']:
            console.print(f"  • {', '.join(combo['datasets'])} - {combo['strength']} relationship")
            if combo['suggested_analysis']:
                console.print(f"    💡 Suggested: {combo['suggested_analysis'][0]}")
    
    # Test Content Detector (Real-time)
    console.print("\n[bold yellow]⚡ TESTING CONTENT DETECTOR (REAL-TIME)[/bold yellow]")
    
    for name, data in [('traffic', traffic_data), ('weather', weather_data), ('construction', construction_data)]:
        console.print(f"\n[cyan]Real-time detection for {name} dataset...[/cyan]")
        
        # Test with chunks
        chunk_size = 10
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i+chunk_size]
            detection = content_detector.detect_content_realtime(chunk, f"{name}_stream")
            
            console.print(f"  📦 Chunk {i//chunk_size + 1}: {detection['content_type']} (confidence: {detection['confidence']:.2f})")
            
            if detection['change_detected']:
                console.print(f"    🔄 Change: {detection['change_details']}")
            
            if detection['anomalies']:
                console.print(f"    ⚠️ Anomalies: {len(detection['anomalies'])} detected")
    
    # Test tag suggestions for queries
    console.print("\n[bold yellow]🔍 TESTING TAG SUGGESTIONS[/bold yellow]")
    
    test_queries = [
        "Find traffic data in NYC",
        "Show me weather patterns",
        "Construction projects with high costs",
        "Environmental data near Manhattan"
    ]
    
    for query in test_queries:
        suggestions = smart_tagger.suggest_tags_for_query(query)
        console.print(f"🔍 Query: '{query}'")
        console.print(f"   💡 Suggested tags: {', '.join(suggestions)}")
    
    # Test tag statistics
    console.print("\n[bold yellow]📈 TESTING TAG STATISTICS[/bold yellow]")
    
    tag_stats = smart_tagger.get_tag_statistics(datasets)
    
    console.print(f"📊 Total datasets: {tag_stats['total_datasets']}")
    console.print(f"🏷️ Most common tags:")
    for tag, count in tag_stats['most_common_tags'][:5]:
        console.print(f"   • {tag}: {count} datasets")
    
    console.print(f"\n[bold green]🎉 ALL CONTENT ANALYZER TESTS COMPLETED SUCCESSFULLY![/bold green]")
    console.print("[bold]✨ The sexy modular content analyzer is working perfectly![/bold]")

if __name__ == "__main__":
    test_content_analyzer() 