#!/usr/bin/env python3
import os
import psutil
import platform

def analyze_system():
    print("=" * 50)
    print("🖥️  SYSTEM ANALYSIS FOR PARALLEL PROCESSING")
    print("=" * 50)
    
    # CPU Analysis
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    
    print(f"🔧 CPU Info:")
    print(f"   Physical cores: {physical_cores}")
    print(f"   Logical cores (threads): {logical_cores}")
    print(f"   Hyperthreading: {'Yes' if logical_cores > physical_cores else 'No'}")
    
    # Memory Analysis
    memory = psutil.virtual_memory()
    print(f"\n💾 Memory Info:")
    print(f"   Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"   Available RAM: {memory.available / (1024**3):.1f} GB")
    print(f"   Current usage: {memory.percent}%")
    
    # Recommendations
    print(f"\n🚀 RECOMMENDED THREAD CONFIGURATION:")
    print(f"   Chunking threads: {min(physical_cores, 8)}")
    print(f"   Embedding threads: {min(logical_cores * 2, 16)}")
    print(f"   Batch size: {max(20, 100 // logical_cores)}")
    
    # Conservative recommendations
    print(f"\n⚡ CONSERVATIVE (SAFE) CONFIGURATION:")
    print(f"   Chunking threads: {max(2, physical_cores // 2)}")
    print(f"   Embedding threads: {logical_cores}")
    print(f"   Batch size: {max(30, 150 // logical_cores)}")
    
    # Aggressive recommendations  
    print(f"\n🔥 AGGRESSIVE (TEST CAREFULLY) CONFIGURATION:")
    print(f"   Chunking threads: {logical_cores}")
    print(f"   Embedding threads: {logical_cores * 3}")
    print(f"   Batch size: {max(15, 50 // logical_cores)}")
    
    return {
        'physical_cores': physical_cores,
        'logical_cores': logical_cores,
        'total_ram_gb': memory.total / (1024**3),
        'available_ram_gb': memory.available / (1024**3)
    }

if __name__ == "__main__":
    analyze_system()