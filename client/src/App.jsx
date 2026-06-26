import React, { useState } from 'react';
import './App.css';

const ASCII_LOGO = `
 ____ ___ _   _ _   _    _    ____  
|  _ \\_ _| \\ | | \\ | |  / \\  |  _ \\ 
| |_) | ||  \\| |  \\| | / _ \\ | |_) |
|  __/| || |\\  | |\\  |/ ___ \\|  _ < 
|_|  |___|_| \\_|_| \\_/_/   \\_\\_| \\_\\
    v2.2 :: ORBITAL SCANNER
`;

function App() {
  const [productId, setProductId] = useState('');
  const [status, setStatus] = useState('idle');
  const [resultImage, setResultImage] = useState(null);
  const [systemLogs, setSystemLogs] = useState(['SYSTEM INITIALIZED...', 'AWAITING TARGET COORDINATES.']);

  const addLog = (msg) => {
    setSystemLogs(prev => [...prev, msg].slice(-15));
  };

  const handleAnalyze = async (e) => {
    e.preventDefault();
    if (!productId) return;
    
    setStatus('processing');
    setResultImage(null);
    addLog(`> INITIATING TARGET ACQUISITION: [${productId}]`);
    
    try {
      const response = await fetch('http://localhost:3001/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ productId }),
      });
      
      const data = await response.json();
      
      if (response.ok && data.status === 'success') {
        setResultImage(data.files[0]);
        setStatus('success');
        addLog(`> SCAN COMPLETE. RENDERED OUTPUT RECEIVED.`);
      } else {
        setStatus('error');
        addLog(`> CRITICAL ERROR: ${data.message || 'Analysis failed'}`);
      }
    } catch (err) {
      setStatus('error');
      addLog('> ERROR: CONNECTION TO UPLINK LOST.');
    }
  };

  return (
    <div className="dashboard-layout">
      
      <header className="ascii-header">
        {ASCII_LOGO}
      </header>

      <form className="search-form" onSubmit={handleAnalyze}>
        <span className="prompt">root@mars-ode:~# ./scan --target </span>
        <input 
          type="text" 
          className="terminal-input"
          placeholder="FRT0000A0A5"
          value={productId}
          onChange={(e) => setProductId(e.target.value.toUpperCase())}
          disabled={status === 'processing'}
          autoFocus
        />
        <button type="submit" className="terminal-btn" disabled={status === 'processing' || !productId}>
          [ EXECUTE ]
        </button>
      </form>

      <div className="layout-grid">
        <aside className="panel">
          <h3>--- SYSTEM LOG ---</h3>
          <br />
          <div className="log-container">
            {systemLogs.map((log, i) => {
              const isError = log.includes('ERROR');
              return (
                <div key={i} className={`log-entry ${isError ? 'error-text' : ''}`}>
                  <span className="log-time">[{new Date().toLocaleTimeString('en-US', { hour12: false })}]</span>
                  <span>{log}</span>
                </div>
              );
            })}
            {status === 'processing' && (
              <div className="log-entry">
                <span className="log-time">[{new Date().toLocaleTimeString('en-US', { hour12: false })}]</span>
                <span className="blink">&gt; DOWNLOADING DATA_CUBE_BLOCK...</span>
              </div>
            )}
          </div>
        </aside>

        <main className="panel map-container">
          {status === 'idle' && (
            <div className="blink">[ AWAITING INPUT ]</div>
          )}
          
          {status === 'processing' && (
            <div>
              <pre>
{`
[================>     ] 78%
ANALYZING HYPERSPECTRAL FREQUENCIES...
CALCULATING MINERAL ABUNDANCES...
`}
              </pre>
            </div>
          )}
          
          {status === 'error' && (
            <div className="error-text">[ FATAL EXCEPTION OCCURRED ]</div>
          )}

          {status === 'success' && resultImage && (
            <>
              <div style={{ position: 'absolute', top: '10px', left: '10px', zIndex: 10, background: '#000' }}>
                [ VIEWPORT :: {productId} ]
              </div>
              {/* Image is filtered in CSS to look like a green phosphor display */}
              <img src={`http://localhost:3001${resultImage}`} alt="Abundance Map" />
            </>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
