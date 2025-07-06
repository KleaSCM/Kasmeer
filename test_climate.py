#!/usr/bin/env python3

import pandas as pd
from src.core.analyzers.environmental_analyzer import EnvironmentalAnalyzer

# Load climate data
df = pd.read_csv('/home/klea/Documents/Dev/AI/DataSets/2023_weather_data_gbs_jjas.csv')

# Analyze with enhanced environmental analyzer
analyzer = EnvironmentalAnalyzer()
result = analyzer.analyze(df)

print('🌡️ CLIMATE DATA ANALYSIS')
print('=' * 60)
print(f"Dataset: {len(df):,} records")
print(f"Columns: {list(df.columns)}")

climate = result['environmental_context']['climate_data']

# Show temperature analysis
if 'tempc_stats' in climate:
    temp_stats = climate['tempc_stats']
    print(f"\n🌡️ TEMPERATURE ANALYSIS:")
    print(f"  • Average: {temp_stats['mean']:.1f}°C")
    print(f"  • Range: {temp_stats['min']:.1f}°C - {temp_stats['max']:.1f}°C")
    print(f"  • Standard Deviation: {temp_stats['std']:.1f}°C")

# Show humidity analysis
if 'rh_stats' in climate:
    hum_stats = climate['rh_stats']
    print(f"\n💧 HUMIDITY ANALYSIS:")
    print(f"  • Average: {hum_stats['mean']:.1f}%")
    print(f"  • Range: {hum_stats['min']:.1f}% - {hum_stats['max']:.1f}%")
    print(f"  • Standard Deviation: {hum_stats['std']:.1f}%")

# Show monthly averages
if 'monthly_averages' in climate:
    monthly_data = climate['monthly_averages']
    print(f"\n📅 MONTHLY AVERAGES:")
    print(f"  • Parameters: {len(monthly_data)}")
    
    for param, monthly_vals in monthly_data.items():
        print(f"  • {param}:")
        for month, value in monthly_vals.items():
            unit = "°C" if "temp" in param.lower() else "%"
            print(f"    - {month}: {value:.1f}{unit}")

# Show summary
print(f"\n📋 SUMMARY:")
for summary_item in result['environmental_context']['summary']:
    print(f"  • {summary_item}")

print("\n" + "=" * 60)
print("✅ Climate analysis complete!") 