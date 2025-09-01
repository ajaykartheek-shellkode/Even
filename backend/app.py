#!/usr/bin/env python3
"""
Flask Backend for Prompt Optimizer MVP.
Serves cached optimization results and provides API endpoints.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from flask import Flask, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
CACHE_DIR = Path("../cache/bedrock")
RESULTS_DIR = Path("../results")
BEST_PROMPT_FILE = Path("../best_prompt_advanced.json")
PROGRESS_FILE = Path("../progress_state.json")


def get_cache_files() -> List[Dict[str, Any]]:
    """Get all cached Bedrock API calls with metadata."""
    if not CACHE_DIR.exists():
        return []
    
    cache_files = []
    for cache_file in CACHE_DIR.glob("*.json"):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            file_stats = cache_file.stat()
            cache_files.append({
                'filename': cache_file.name,
                'hash': cache_file.stem,
                'created_at': file_stats.st_ctime,
                'size_kb': round(file_stats.st_size / 1024, 2),
                'model_id': data.get('model_id', 'unknown'),
                'temperature': data.get('temperature', 0),
                'max_tokens': data.get('max_tokens', 0),
                'prompt_length': len(data.get('prompt', '')),
                'output_length': len(data.get('output', '')),
                'response_time': data.get('response_time', 0)
            })
        except Exception as e:
            print(f"Error reading cache file {cache_file}: {e}")
            continue
    
    # Sort by creation time (newest first)
    cache_files.sort(key=lambda x: x['created_at'], reverse=True)
    return cache_files


def get_optimization_results() -> Optional[Dict[str, Any]]:
    """Load the best prompt optimization results."""
    if not BEST_PROMPT_FILE.exists():
        return None
    
    try:
        with open(BEST_PROMPT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading best prompt results: {e}")
        return None


def format_timestamp(timestamp: float) -> str:
    """Format Unix timestamp to readable string."""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def get_optimization_progress() -> Optional[Dict[str, Any]]:
    """Load current optimization progress."""
    if not PROGRESS_FILE.exists():
        return None
    
    try:
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading progress file: {e}")
        return None


@app.route('/')
def index():
    """Health check endpoint."""
    return jsonify({
        'status': 'running',
        'service': 'Prompt Optimizer MVP Backend',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/cache/stats')
def cache_stats():
    """Get cache statistics and overview."""
    cache_files = get_cache_files()
    
    if not cache_files:
        return jsonify({
            'total_files': 0,
            'total_size_mb': 0,
            'avg_response_time': 0,
            'models_used': [],
            'cache_directory': str(CACHE_DIR.absolute())
        })
    
    total_size = sum(f['size_kb'] for f in cache_files) / 1024  # Convert to MB
    avg_response_time = sum(f['response_time'] for f in cache_files) / len(cache_files)
    models_used = list(set(f['model_id'] for f in cache_files))
    
    return jsonify({
        'total_files': len(cache_files),
        'total_size_mb': round(total_size, 2),
        'avg_response_time': round(avg_response_time, 2),
        'models_used': models_used,
        'cache_directory': str(CACHE_DIR.absolute()),
        'oldest_entry': format_timestamp(min(f['created_at'] for f in cache_files)),
        'newest_entry': format_timestamp(max(f['created_at'] for f in cache_files))
    })


@app.route('/api/cache/files')
def cache_files():
    """Get detailed list of all cached files."""
    cache_files = get_cache_files()
    
    # Format timestamps for display
    for file_info in cache_files:
        file_info['created_at_formatted'] = format_timestamp(file_info['created_at'])
    
    return jsonify({
        'files': cache_files,
        'total_count': len(cache_files)
    })


@app.route('/api/cache/file/<hash_id>')
def get_cache_file(hash_id: str):
    """Get specific cached file contents by hash ID."""
    cache_file = CACHE_DIR / f"{hash_id}.json"
    
    if not cache_file.exists():
        return jsonify({'error': 'Cache file not found'}), 404
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return jsonify({
            'hash_id': hash_id,
            'data': data,
            'file_path': str(cache_file.absolute())
        })
    except Exception as e:
        return jsonify({'error': f'Error reading cache file: {str(e)}'}), 500


@app.route('/api/optimization/results')
def optimization_results():
    """Get the best prompt optimization results."""
    results = get_optimization_results()
    
    if not results:
        return jsonify({'error': 'No optimization results found'}), 404
    
    # Add formatted timestamps
    if 'metadata' in results and 'created_at' in results['metadata']:
        results['metadata']['created_at_formatted'] = format_timestamp(
            results['metadata']['created_at']
        )
    
    return jsonify(results)


@app.route('/api/optimization/summary')
def optimization_summary():
    """Get optimization summary statistics."""
    results = get_optimization_results()
    
    if not results:
        return jsonify({'error': 'No optimization results found'}), 404
    
    opt_results = results.get('optimization_results', {})
    metadata = results.get('metadata', {})
    
    return jsonify({
        'best_score': opt_results.get('best_score', 0),
        'optimization_time_seconds': opt_results.get('optimization_time', 0),
        'optimization_time_formatted': f"{opt_results.get('optimization_time', 0):.1f} seconds",
        'n_trials': opt_results.get('n_trials', 0),
        'training_examples': metadata.get('training_examples', 0),
        'validation_examples': metadata.get('validation_examples', 0),
        'region': metadata.get('region', 'unknown'),
        'created_at': format_timestamp(metadata.get('created_at', 0)),
        'best_params': opt_results.get('best_params', {}),
        'model_performance': {
            'accuracy_percent': round(opt_results.get('best_score', 0) * 100, 1),
            'performance_tier': (
                'Excellent' if opt_results.get('best_score', 0) > 0.95 else
                'Good' if opt_results.get('best_score', 0) > 0.85 else
                'Fair' if opt_results.get('best_score', 0) > 0.75 else
                'Poor'
            )
        }
    })


@app.route('/api/optimization/prompt')
def get_optimized_prompt():
    """Get the optimized prompt configuration and text."""
    results = get_optimization_results()
    
    if not results:
        return jsonify({'error': 'No optimization results found'}), 404
    
    return jsonify({
        'prompt_configuration': results.get('prompt_configuration', {}),
        'instruction_text': results.get('instruction_text', ''),
        'usage_instructions': results.get('usage_instructions', {}),
        'best_params': results.get('optimization_results', {}).get('best_params', {})
    })


@app.route('/api/optimization/progress')
def optimization_progress():
    """Get current optimization progress."""
    progress = get_optimization_progress()
    
    if not progress:
        return jsonify({
            'status': 'idle',
            'message': 'No optimization currently running'
        })
    
    # Calculate additional metrics
    current_time = time.time()
    start_time = progress.get('start_time', current_time)
    elapsed_time = current_time - start_time
    
    current_trial = progress.get('current_trial', 0)
    total_trials = progress.get('total_trials', 1)
    
    # Estimate time remaining
    if current_trial > 0:
        avg_time_per_trial = elapsed_time / current_trial
        remaining_trials = total_trials - current_trial
        estimated_remaining = avg_time_per_trial * remaining_trials
    else:
        estimated_remaining = 0
    
    return jsonify({
        'status': progress.get('status', 'unknown'),
        'current_trial': current_trial,
        'total_trials': total_trials,
        'progress_percent': round((current_trial / total_trials) * 100, 1) if total_trials > 0 else 0,
        'best_score_so_far': progress.get('best_score_so_far', 0),
        'current_score': progress.get('current_score', 0),
        'elapsed_time_seconds': elapsed_time,
        'estimated_remaining_seconds': estimated_remaining,
        'elapsed_time_formatted': f"{elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s",
        'estimated_remaining_formatted': f"{estimated_remaining // 60:.0f}m {estimated_remaining % 60:.0f}s",
        'api_calls_made': progress.get('api_calls_made', 0),
        'estimated_cost': progress.get('api_calls_made', 0) * 0.15,
        'last_updated': progress.get('last_updated', current_time),
        'current_phase': progress.get('current_phase', 'Unknown')
    })


@app.route('/api/health')
def health_check():
    """Detailed health check with system information."""
    cache_exists = CACHE_DIR.exists()
    results_exist = BEST_PROMPT_FILE.exists()
    cache_file_count = len(list(CACHE_DIR.glob("*.json"))) if cache_exists else 0
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'system_info': {
            'cache_directory_exists': cache_exists,
            'cache_directory': str(CACHE_DIR.absolute()),
            'cache_file_count': cache_file_count,
            'results_file_exists': results_exist,
            'results_file': str(BEST_PROMPT_FILE.absolute())
        },
        'endpoints': {
            'cache_stats': '/api/cache/stats',
            'cache_files': '/api/cache/files',
            'optimization_results': '/api/optimization/results',
            'optimization_summary': '/api/optimization/summary',
            'optimized_prompt': '/api/optimization/prompt',
            'optimization_progress': '/api/optimization/progress'
        }
    })


if __name__ == '__main__':
    print("=== Prompt Optimizer MVP Backend ===")
    print(f"Cache directory: {CACHE_DIR.absolute()}")
    print(f"Results file: {BEST_PROMPT_FILE.absolute()}")
    print("Starting Flask development server...")
    print("Access API at: http://localhost:5001")
    
    # Check if cache and results exist
    if CACHE_DIR.exists():
        cache_count = len(list(CACHE_DIR.glob("*.json")))
        print(f"Found {cache_count} cached API calls")
    else:
        print("Cache directory not found")
    
    if BEST_PROMPT_FILE.exists():
        print("Optimization results found")
    else:
        print("No optimization results found")
    
    app.run(debug=True, host='0.0.0.0', port=5001)