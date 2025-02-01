// Global state
let currentSession = null;
let outputEventSource = null;

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const configSection = document.getElementById('configSection');
const outputSection = document.getElementById('outputSection');
const analysisForm = document.getElementById('analysisForm');
const outputText = document.getElementById('outputText');
const commandPreview = document.getElementById('commandPreview');
const downloadResults = document.getElementById('downloadResults');
const newAnalysis = document.getElementById('newAnalysis');
const clientSelect = document.getElementById('client');
const ollamaSettings = document.getElementById('ollamaSettings');
const openaiSettings = document.getElementById('openaiSettings');

// Event Listeners
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', handleDragOver);
dropZone.addEventListener('dragleave', handleDragLeave);
dropZone.addEventListener('drop', handleDrop);
fileInput.addEventListener('change', handleFileSelect);
analysisForm.addEventListener('submit', handleAnalysis);
analysisForm.addEventListener('input', updateCommandPreview);
clientSelect.addEventListener('change', toggleClientSettings);
downloadResults.addEventListener('click', downloadAnalysisResults);
newAnalysis.addEventListener('click', resetUI);

// File Upload Handlers
function handleDragOver(e) {
    e.preventDefault();
    dropZone.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (isValidVideoFile(file)) {
        handleFile(file);
    } else {
        alert('Please upload a valid video file (MP4, AVI, MOV, or MKV)');
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file && isValidVideoFile(file)) {
        handleFile(file);
    }
}

function isValidVideoFile(file) {
    const validTypes = ['.mp4', '.avi', '.mov', '.mkv'];
    return validTypes.some(type => file.name.toLowerCase().endsWith(type));
}

async function handleFile(file) {
    const formData = new FormData();
    formData.append('video', file);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        if (response.ok) {
            currentSession = data.session_id;
            showConfigSection();
        } else {
            throw new Error(data.error || 'Upload failed');
        }
    } catch (error) {
        alert(`Error uploading file: ${error.message}`);
    }
}

// Analysis Handlers
async function handleAnalysis(e) {
    e.preventDefault();
    if (!currentSession) return;
    
    const formData = new FormData(analysisForm);
    showOutputSection();
    
    // Close any existing event source
    if (outputEventSource) {
        outputEventSource.close();
    }
    
    // Clear previous output and show loading
    outputText.textContent = '';
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading';
    loadingDiv.innerHTML = '<div class="loading-text">Analyzing video...</div>';
    outputText.parentElement.appendChild(loadingDiv);
    
    try {
        // Make POST request to start analysis
        const response = await fetch(`/analyze/${currentSession}`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.error || 'Failed to start analysis');
        }
    } catch (error) {
        outputText.textContent = `Error starting analysis: ${error.message}\n`;
        document.querySelector('.output-actions').style.display = 'flex';
        downloadResults.style.display = 'none';
        loadingDiv.remove();
        return;
    }

    // Start SSE connection for output
    outputEventSource = new EventSource(`/analyze/${currentSession}/stream`);
    
    outputEventSource.onmessage = (event) => {
        // Remove loading indicator on first message
        if (loadingDiv.parentElement) {
            loadingDiv.remove();
        }
        
        outputText.textContent += event.data + '\n';
        outputText.scrollTop = outputText.scrollHeight;
        
        if (event.data.includes('Analysis completed successfully')) {
            outputEventSource.close();
            document.querySelector('.output-actions').style.display = 'flex';
            downloadResults.style.display = 'inline-block';
        } else if (event.data.includes('Analysis failed')) {
            outputEventSource.close();
            outputText.textContent += '\nAnalysis failed. Please check the output above for errors.\n';
            // Show new analysis button but not download button
            document.querySelector('.output-actions').style.display = 'flex';
            downloadResults.style.display = 'none';
        }
    };
    
    outputEventSource.onerror = (error) => {
        console.error('SSE Error:', error);
        outputEventSource.close();
        
        // Remove loading indicator if it exists
        if (loadingDiv.parentElement) {
            loadingDiv.remove();
        }
        
        outputText.textContent += '\nError: Connection to server lost. Please try again.\n';
        // Show new analysis button but not download button
        document.querySelector('.output-actions').style.display = 'flex';
        downloadResults.style.display = 'none';
    };
}

// UI Updates
function showConfigSection() {
    dropZone.style.display = 'none';
    configSection.style.display = 'block';
    updateCommandPreview();
}

function showOutputSection() {
    configSection.style.display = 'none';
    outputSection.style.display = 'block';
    document.querySelector('.output-actions').style.display = 'none';
}

function toggleClientSettings() {
    const client = clientSelect.value;
    if (client === 'ollama') {
        ollamaSettings.style.display = 'block';
        openaiSettings.style.display = 'none';
    } else {
        ollamaSettings.style.display = 'none';
        openaiSettings.style.display = 'block';
    }
    updateCommandPreview();
}

function updateCommandPreview() {
    const formData = new FormData(analysisForm);
    let command = 'video-analyzer <video_path>';
    
    for (const [key, value] of formData.entries()) {
        if (value) {
            if (key === 'keep-frames') {
                command += ` --${key}`;
            } else {
                command += ` --${key} ${value}`;
            }
        }
    }
    
    commandPreview.textContent = command;
}

// Results Handling
async function downloadAnalysisResults() {
    if (!currentSession) return;
    
    try {
        const response = await fetch(`/results/${currentSession}`);
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.error || 'Failed to fetch results');
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'analysis.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    } catch (error) {
        alert(`Error downloading results: ${error.message}`);
        console.error('Download error:', error);
    }
}

function resetUI() {
    // Clean up current session
    if (currentSession) {
        fetch(`/cleanup/${currentSession}`, { method: 'POST' })
            .catch(error => console.error('Cleanup error:', error));
    }
    
    // Reset state
    currentSession = null;
    if (outputEventSource) {
        outputEventSource.close();
    }
    
    // Reset form
    analysisForm.reset();
    
    // Reset UI
    dropZone.style.display = 'block';
    configSection.style.display = 'none';
    outputSection.style.display = 'none';
    outputText.textContent = '';
    fileInput.value = '';
    
    // Reset client settings
    toggleClientSettings();
}

// Initialize UI
toggleClientSettings();
