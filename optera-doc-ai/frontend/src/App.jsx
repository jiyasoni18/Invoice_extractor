import { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [taskId, setTaskId] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) setFile(droppedFile);
  };

  const handleFileChange = (e) => {
    if (e.target.files[0]) setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    setIsProcessing(true);
    setResult(null);
    setError(null);
    setTaskId(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const res = await fetch(`${apiUrl}/upload`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error('Failed to upload image');
      const data = await res.json();
      setTaskId(data.task_id);
    } catch (err) {
      setError(err.message);
      setIsProcessing(false);
    }
  };

  useEffect(() => {
    let intervalId;
    
    if (taskId && isProcessing) {
      intervalId = setInterval(async () => {
        try {
          const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
          const res = await fetch(`${apiUrl}/status/${taskId}`);
          if (!res.ok) throw new Error("Failed to fetch status");
          const data = await res.json();
          
          setResult(data);
          
          if (data.done) {
            setIsProcessing(false);
            clearInterval(intervalId);
          }
        } catch (err) {
          console.error(err);
          setError("Error fetching status from backend.");
          setIsProcessing(false);
          clearInterval(intervalId);
        }
      }, 1000);
    }
    
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [taskId, isProcessing]);

  return (
    <div className="app-container">
      <header className="header">
        <h1>Optera Document AI</h1>
        <p>Real-Time Cost-Optimized Document Pipeline</p>
      </header>

      <main className="main-content">
        <section className="upload-section">
          <form onSubmit={handleSubmit}>
            <div 
              className={`drop-zone ${file ? 'has-file' : ''}`}
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => document.getElementById('file-input').click()}
            >
              <input 
                id="file-input"
                type="file" 
                accept="image/*,.pdf" 
                onChange={handleFileChange} 
                hidden 
              />
              {file ? (
                <div className="file-info">
                  <span className="file-icon">📄</span>
                  <p>{file.name}</p>
                </div>
              ) : (
                <div className="upload-prompt">
                  <span className="upload-icon">☁️</span>
                  <p>Drag & drop an image or PDF here or click to browse</p>
                  <small>Supports: Invoices, Mechanic Logs, Meters, PDFs</small>
                </div>
              )}
            </div>
            
            <button 
              type="submit" 
              className={`submit-btn ${isProcessing ? 'processing' : ''}`}
              disabled={!file || isProcessing}
            >
              {isProcessing ? 'Processing Pipeline...' : 'Extract JSON'}
            </button>
          </form>
        </section>

        {error && <div className="error-message">{error}</div>}

        {result && (
          <section className="results-section">
            <div className="stages-container">
              <h3>Live Processing Pipeline</h3>
              <ul className="timeline">
                {result.stages.map((stage, idx) => (
                  <li key={idx} className={`timeline-item ${stage.status}`}>
                    <div className="timeline-marker"></div>
                    <div className="timeline-content">
                      <h4>{stage.name}</h4>
                      <p>{stage.details}</p>
                    </div>
                  </li>
                ))}
                
                {/* Show a loading spinner for the next stage if not done */}
                {!result.done && (
                  <li className="timeline-item pending">
                    <div className="timeline-marker pulse-marker"></div>
                    <div className="timeline-content">
                      <h4>Working...</h4>
                    </div>
                  </li>
                )}
              </ul>
            </div>

            {result.final_json && (
              <div className="json-container">
                <h3>Final Extracted Output</h3>
                <pre>
                  <code>{JSON.stringify(result.final_json, null, 2)}</code>
                </pre>
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
