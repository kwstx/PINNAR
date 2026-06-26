const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

// Ensure the outputs directory exists so static files can be served and python can write
const outputsDir = path.join(__dirname, 'public', 'outputs');
if (!fs.existsSync(outputsDir)) {
    fs.mkdirSync(outputsDir, { recursive: true });
}

// Serve static files (like generated images)
app.use('/outputs', express.static(outputsDir));

app.post('/api/analyze', (req, res) => {
    const { productId } = req.body;
    
    if (!productId) {
        return res.status(400).json({ error: 'Product ID is required' });
    }

    // Path to the python inference script
    const scriptPath = path.join(__dirname, '..', 'src', 'inference.py');
    
    console.log(`Starting analysis for product: ${productId}`);
    
    // Spawn the python process
    // Use the python from the active environment if possible. 
    // Usually 'python' is sufficient if the venv is activated.
    const pythonProcess = spawn('python', [scriptPath, productId, outputsDir]);
    
    let resultData = '';
    let errorData = '';

    // Collect data from standard output
    pythonProcess.stdout.on('data', (data) => {
        const text = data.toString();
        resultData += text;
        // Optionally, parse JSON progress updates here and emit via WebSockets or SSE if needed.
        console.log(`[Python]: ${text.trim()}`);
    });

    // Collect data from standard error
    pythonProcess.stderr.on('data', (data) => {
        errorData += data.toString();
        console.error(`[Python Error]: ${data.toString().trim()}`);
    });

    // Handle process completion
    pythonProcess.on('close', (code) => {
        if (code !== 0) {
            console.error(`Python process exited with code ${code}`);
            return res.status(500).json({ 
                status: 'error', 
                message: 'Inference pipeline failed', 
                details: errorData 
            });
        }
        
        try {
            // Find the final JSON result string from stdout
            // Since python prints progress JSONs too, we grab the last valid JSON output
            const outputLines = resultData.trim().split('\n');
            let finalResult = null;
            
            for (let i = outputLines.length - 1; i >= 0; i--) {
                try {
                    const parsed = JSON.parse(outputLines[i]);
                    // If it contains the 'files' array, it's the final result
                    if (parsed && typeof parsed === 'object' && 'files' in parsed) {
                        finalResult = parsed;
                        break;
                    }
                } catch (e) {
                    // Ignore lines that aren't JSON
                }
            }
            
            if (finalResult) {
                // Adjust file paths to be relative to the server so frontend can fetch them
                finalResult.files = finalResult.files.map(file => {
                    const filename = path.basename(file);
                    return `/outputs/${filename}`;
                });
                return res.json(finalResult);
            } else {
                return res.status(500).json({ status: 'error', message: 'Failed to parse python output' });
            }
        } catch (err) {
            return res.status(500).json({ status: 'error', message: err.message, raw: resultData });
        }
    });
});

app.listen(PORT, () => {
    console.log(`PINNAR backend API running on http://localhost:${PORT}`);
});
