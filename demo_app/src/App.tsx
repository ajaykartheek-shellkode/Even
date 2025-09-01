import React, { useState, useEffect } from 'react';
import './App.css';

interface OptimizationSummary {
  best_score: number;
  optimization_time_formatted: string;
  optimization_time_seconds: number;
  n_trials: number;
  training_examples: number;
  validation_examples: number;
  region: string;
  created_at: string;
  best_params: {
    instruction_idx: number;
    use_fs: boolean;
    k: number;
    temperature: number;
    max_tokens: number;
  };
  model_performance: {
    accuracy_percent: number;
    performance_tier: string;
  };
}

interface CacheStats {
  total_files: number;
  total_size_mb: number;
  avg_response_time: number;
  models_used: string[];
  cache_directory: string;
  oldest_entry?: string;
  newest_entry?: string;
}

interface HealthCheck {
  status: string;
  system_info: {
    cache_directory_exists: boolean;
    cache_file_count: number;
    results_file_exists: boolean;
  };
}

interface OptimizationProgress {
  status: 'idle' | 'running' | 'completed' | 'failed';
  current_trial: number;
  total_trials: number;
  progress_percent: number;
  best_score_so_far: number;
  current_score: number;
  elapsed_time_formatted: string;
  estimated_remaining_formatted: string;
  api_calls_made: number;
  estimated_cost: number;
  current_phase: string;
  message?: string;
}

const App: React.FC = () => {
  const [optimizationData, setOptimizationData] = useState<OptimizationSummary | null>(null);
  const [cacheData, setCacheData] = useState<CacheStats | null>(null);
  const [healthData, setHealthData] = useState<HealthCheck | null>(null);
  const [progressData, setProgressData] = useState<OptimizationProgress | null>(null);
  const [optimizedPrompt, setOptimizedPrompt] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<string>('');

  useEffect(() => {
    const fetchData = async (isRefresh = false) => {
      try {
        if (!isRefresh) setLoading(true);
        else setRefreshing(true);
        
        // Fetch optimization summary
        const optimizationResponse = await fetch('http://localhost:5001/api/optimization/summary');
        if (optimizationResponse.ok) {
          const optData = await optimizationResponse.json();
          setOptimizationData(optData);
          
          // Fetch optimized prompt only if optimization data exists
          const promptResponse = await fetch('http://localhost:5001/api/optimization/prompt');
          if (promptResponse.ok) {
            const promptData = await promptResponse.json();
            setOptimizedPrompt(promptData.instruction_text || '');
          }
        } else {
          // No optimization data available - set demo/placeholder data
          setOptimizationData(null);
          setOptimizedPrompt('');
        }

        // Always fetch cache stats and health (these work without optimization)
        const cacheResponse = await fetch('http://localhost:5001/api/cache/stats');
        if (cacheResponse.ok) {
          const cacheStatsData = await cacheResponse.json();
          setCacheData(cacheStatsData);
        }

        const healthResponse = await fetch('http://localhost:5001/api/health');
        if (healthResponse.ok) {
          const healthCheckData = await healthResponse.json();
          setHealthData(healthCheckData);
        }

        // Always fetch progress data
        const progressResponse = await fetch('http://localhost:5001/api/optimization/progress');
        if (progressResponse.ok) {
          const progressInfo = await progressResponse.json();
          console.log('📊 Progress data:', progressInfo);
          setProgressData(progressInfo);
        } else {
          console.error('❌ Failed to fetch progress data');
        }

        setLastUpdate(new Date().toLocaleTimeString());
        setLoading(false);
        setRefreshing(false);
      } catch (err) {
        setError(`Failed to fetch data: ${err}`);
        setLoading(false);
        setRefreshing(false);
      }
    };

    // Initial fetch
    fetchData();
    
    // Fixed polling interval - start simple
    const interval = setInterval(() => {
      console.log('🔄 Fetching data...', new Date().toLocaleTimeString());
      fetchData(true);
    }, 5000); // Every 5 seconds for testing
    
    // Cleanup interval on unmount
    return () => {
      clearInterval(interval);
    };
  }, []);

  const getPerformanceColor = (tier: string): string => {
    switch (tier) {
      case 'Excellent': return '#22c55e';
      case 'Good': return '#3b82f6';
      case 'Fair': return '#f59e0b';
      case 'Poor': return '#ef4444';
      default: return '#6b7280';
    }
  };

  if (loading) {
    return (
      <div className="app">
        <div className="loading">
          <div className="spinner"></div>
          <p>Loading optimization results...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="app">
        <div className="error">
          <h2>⚠️ Error</h2>
          <p>{error}</p>
          <button onClick={() => window.location.reload()}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="header">
        <h1>🚀 Prompt Optimizer MVP</h1>
        <p>Medical Invoice Extraction Optimization Results</p>
        <small style={{opacity: 0.7}}>
          {refreshing && <span style={{color: '#fbbf24'}}>🔄 Refreshing...</span>}
          {!refreshing && lastUpdate && <span>Last updated: {lastUpdate}</span>}
          {!refreshing && !lastUpdate && <span>Loading...</span>}
          {progressData?.status === 'running' && <span style={{color: '#22c55e', marginLeft: '10px'}}>⚡ Live updates every 3s</span>}
        </small>
      </header>

      {/* Live Optimization Progress */}
      {progressData?.status === 'running' && (
        <div className="card progress-card" style={{background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white', marginBottom: '24px'}}>
          <h2 style={{color: 'white', margin: '0 0 20px 0'}}>⚡ Optimization In Progress</h2>
          
          {/* Progress Bar */}
          <div style={{background: 'rgba(255,255,255,0.2)', borderRadius: '10px', padding: '4px', marginBottom: '20px'}}>
            <div 
              style={{
                width: `${progressData.progress_percent}%`,
                height: '20px',
                background: 'linear-gradient(90deg, #22c55e, #16a34a)',
                borderRadius: '6px',
                transition: 'width 0.5s ease',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
                fontSize: '12px',
                fontWeight: 'bold'
              }}
            >
              {progressData.progress_percent}%
            </div>
          </div>

          {/* Progress Details */}
          <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px'}}>
            <div>
              <div style={{opacity: 0.8, fontSize: '14px'}}>Trial Progress</div>
              <div style={{fontSize: '18px', fontWeight: 'bold'}}>
                {progressData.current_trial} of {progressData.total_trials}
              </div>
            </div>
            <div>
              <div style={{opacity: 0.8, fontSize: '14px'}}>Best Score So Far</div>
              <div style={{fontSize: '18px', fontWeight: 'bold', color: '#22c55e'}}>
                {(progressData.best_score_so_far * 100).toFixed(1)}%
              </div>
            </div>
            <div>
              <div style={{opacity: 0.8, fontSize: '14px'}}>Time Elapsed</div>
              <div style={{fontSize: '18px', fontWeight: 'bold'}}>
                {progressData.elapsed_time_formatted}
              </div>
            </div>
            <div>
              <div style={{opacity: 0.8, fontSize: '14px'}}>Est. Remaining</div>
              <div style={{fontSize: '18px', fontWeight: 'bold'}}>
                {progressData.estimated_remaining_formatted}
              </div>
            </div>
          </div>

          {/* Current Phase */}
          <div style={{marginTop: '16px', padding: '12px', background: 'rgba(255,255,255,0.1)', borderRadius: '8px'}}>
            <div style={{opacity: 0.8, fontSize: '14px', marginBottom: '4px'}}>Current Phase</div>
            <div style={{fontWeight: '500'}}>{progressData.current_phase}</div>
          </div>

          {/* Cost Tracking */}
          <div style={{marginTop: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '14px', opacity: 0.9}}>
            <span>💰 API Calls: {progressData.api_calls_made}</span>
            <span>💸 Est. Cost: ${progressData.estimated_cost.toFixed(2)}</span>
          </div>
        </div>
      )}

      {/* No Optimization State */}
      {!optimizationData && (
        <div className="card health-card">
          <h2>🎯 Ready to Optimize</h2>
          <div style={{textAlign: 'center', padding: '40px 20px'}}>
            <div style={{fontSize: '4rem', marginBottom: '20px'}}>🚀</div>
            <h3 style={{color: '#1f2937', marginBottom: '16px'}}>No optimization runs yet</h3>
            <p style={{color: '#6b7280', marginBottom: '24px', lineHeight: '1.6'}}>
              Run the prompt optimizer to see performance metrics, winning configurations, and cost savings.
            </p>
            <div style={{background: '#f8f9fa', padding: '16px', borderRadius: '8px', marginBottom: '20px'}}>
              <code style={{color: '#374151', fontSize: '14px'}}>
                python prompt_optimizer.py --train-folder data/train --val-folder data/val --trials 5
              </code>
            </div>
            <p style={{color: '#6b7280', fontSize: '14px'}}>
              💡 Results will appear here automatically once optimization completes
            </p>
          </div>
        </div>
      )}

      {/* Optimization Insights */}
      {healthData && optimizationData && (
        <div className="card health-card">
          <h2>🔍 Optimization Insights</h2>
          <div className="health-grid">
            <div className="health-item healthy">
              <span>⚡ Total Trials Run</span>
              <span>{optimizationData.n_trials} trials</span>
            </div>
            <div className="health-item healthy">
              <span>⏱️ Time Invested</span>
              <span>{Math.round(optimizationData.optimization_time_seconds / 60)} minutes</span>
            </div>
            <div className="health-item healthy">
              <span>💰 API Calls Cached</span>
              <span>{healthData.system_info.cache_file_count} (cost savings)</span>
            </div>
            <div className="health-item healthy">
              <span>📊 Data Quality</span>
              <span>{optimizationData.training_examples} train + {optimizationData.validation_examples} val</span>
            </div>
          </div>
        </div>
      )}

      {/* Optimization Results */}
      {optimizationData && (
        <div className="results-grid">
          <div className="card performance-card">
            <h2>📈 Model Performance</h2>
            <div className="performance-score">
              <span className="score" style={{ color: getPerformanceColor(optimizationData.model_performance.performance_tier) }}>
                {optimizationData.model_performance.accuracy_percent}%
              </span>
              <span className="tier" style={{ color: getPerformanceColor(optimizationData.model_performance.performance_tier) }}>
                {optimizationData.model_performance.performance_tier}
              </span>
            </div>
          </div>

          <div className="card details-card">
            <h2>📈 Performance Breakdown</h2>
            <div className="details-grid">
              <div className="detail-item">
                <span className="label">Success Rate:</span>
                <span className="value">{optimizationData.model_performance.performance_tier} ({optimizationData.model_performance.accuracy_percent}%)</span>
              </div>
              <div className="detail-item">
                <span className="label">Efficiency:</span>
                <span className="value">{Math.round((optimizationData.model_performance.accuracy_percent / optimizationData.n_trials) * 10) / 10}% per trial</span>
              </div>
              <div className="detail-item">
                <span className="label">Data Utilization:</span>
                <span className="value">{Math.round((optimizationData.validation_examples / optimizationData.training_examples) * 100)}% validation ratio</span>
              </div>
              <div className="detail-item">
                <span className="label">Time per Trial:</span>
                <span className="value">{Math.round(optimizationData.optimization_time_seconds / optimizationData.n_trials)} seconds</span>
              </div>
              <div className="detail-item">
                <span className="label">AWS Region:</span>
                <span className="value">{optimizationData.region}</span>
              </div>
              <div className="detail-item">
                <span className="label">Last Run:</span>
                <span className="value">{new Date(optimizationData.created_at).toLocaleDateString()}</span>
              </div>
            </div>
          </div>

          <div className="card params-card">
            <h2>🎯 Winning Configuration</h2>
            <div className="params-grid">
              <div className="param-item">
                <span className="param-label">Instruction Style:</span>
                <span className="param-value">
                  {(() => {
                    if (optimizationData.best_params.instruction_idx === 0) return 'Strict (Detailed)';
                    if (optimizationData.best_params.instruction_idx === 1) return 'Concise (Brief)';
                    return 'Safe Fallback';
                  })()}
                </span>
              </div>
              <div className="param-item">
                <span className="param-label">Learning Method:</span>
                <span className="param-value">
                  {optimizationData.best_params.use_fs ? `Few-shot (${optimizationData.best_params.k} examples)` : 'Zero-shot'}
                </span>
              </div>
              <div className="param-item">
                <span className="param-label">Creativity Level:</span>
                <span className="param-value">
                  {(() => {
                    const temp = optimizationData.best_params.temperature;
                    if (temp === 0) return 'Deterministic (0.0)';
                    if (temp <= 0.1) return 'Very Low';
                    if (temp <= 0.3) return 'Low';
                    return 'Medium';
                  })()}
                </span>
              </div>
              <div className="param-item">
                <span className="param-label">Response Capacity:</span>
                <span className="param-value">
                  {(() => {
                    const tokens = optimizationData.best_params.max_tokens;
                    let size = 'Compact';
                    if (tokens >= 5000) size = 'Large';
                    else if (tokens >= 3000) size = 'Medium';
                    return `${size} (${tokens.toLocaleString()} tokens)`;
                  })()}
                </span>
              </div>
              <div className="param-item">
                <span className="param-label">Cost Impact:</span>
                <span className="param-value">
                  {(() => {
                    const tokens = optimizationData.best_params.max_tokens;
                    const k = optimizationData.best_params.k;
                    if (tokens <= 2000 && k <= 2) return 'Low cost';
                    if (tokens <= 5000) return 'Medium cost';
                    return 'High cost';
                  })()}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Optimized Prompt Display */}
      {optimizedPrompt && (
        <div className="card prompt-card">
          <h2>📝 Optimized Prompt</h2>
          <div className="prompt-display">
            <pre style={{
              background: '#f8f9fa',
              padding: '16px',
              borderRadius: '8px',
              overflow: 'auto',
              maxHeight: '400px',
              fontSize: '12px',
              lineHeight: '1.4'
            }}>
              {optimizedPrompt}
            </pre>
          </div>
          <div style={{marginTop: '12px', color: '#6b7280', fontSize: '14px'}}>
            💡 This is the winning prompt configuration that achieved {optimizationData?.model_performance.accuracy_percent}% accuracy
          </div>
        </div>
      )}

      {/* Cache Information - Always show */}
      {cacheData && (
        <div className="card cache-card">
          <h2>💾 System Cache</h2>
          <div className="cache-grid">
            <div className="cache-item">
              <span className="cache-label">Cached API Calls:</span>
              <span className="cache-value">{cacheData.total_files} calls</span>
            </div>
            <div className="cache-item">
              <span className="cache-label">Storage Used:</span>
              <span className="cache-value">{cacheData.total_size_mb} MB</span>
            </div>
            <div className="cache-item">
              <span className="cache-label">Cache Directory:</span>
              <span className="cache-value">Ready</span>
            </div>
            <div className="cache-item">
              <span className="cache-label">Status:</span>
              <span className="cache-value">Active</span>
            </div>
          </div>
          {cacheData.total_files === 0 && !optimizationData && (
            <div className="cache-notice">
              <p>💡 Cache is ready! API calls will be automatically cached during optimization to save costs.</p>
            </div>
          )}
          {cacheData.total_files === 0 && optimizationData && (
            <div className="cache-notice">
              <p>💡 No cached API calls yet. Run a new optimization to see cost savings!</p>
            </div>
          )}
          {cacheData.total_files > 0 && (
            <div className="cache-notice" style={{background: 'rgba(34, 197, 94, 0.1)', borderColor: 'rgba(34, 197, 94, 0.3)'}}>
              <p>✅ Cache is working! Saved ~${((cacheData.total_files - 1) * 0.15).toFixed(2)} in API costs.</p>
            </div>
          )}
        </div>
      )}

      {/* Footer */}
      <footer className="footer">
        <p>🤖 Powered by AWS Bedrock • Built with React & Flask</p>
      </footer>
    </div>
  );
};

export default App;