const API_BASE_URL = window.location.origin;

let currentFile = null;
let analysisAbortController = null;
let spCredentials = {};
let isSharePointFile = false;
let progressSource = null;

document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

function initApp() {
    initEventListeners();
    showScreen('initialScreen');
}

function initEventListeners() {
    // File upload/drop handlers
    document.getElementById('dropZone').addEventListener('click', () => {
        document.getElementById('videoInput').click();
    });

    document.getElementById('videoInput').addEventListener('change', handleFileSelect);
    
    const dropZone = document.getElementById('dropZone');
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('drop', handleDrop);
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

    // Navigation buttons
    document.querySelectorAll('[data-screen]').forEach(btn => {
        btn.addEventListener('click', () => {
            if (analysisAbortController) {
                cancelAnalysis().finally(() => showScreen(btn.dataset.screen));
            } else {
                showScreen(btn.dataset.screen);
            }
        });
    });

    document.querySelector('#resultsScreen .btn.secondary').addEventListener('click', handleNewAnalysis);

    // Analysis controls
    document.getElementById('analyzeBtn').addEventListener('click', startAnalysis);
    document.getElementById('cancelBtn').addEventListener('click', cancelAnalysis);
    document.getElementById('downloadBtn').addEventListener('click', () => {
        // Handled in setupDownload
    });
    document.getElementById('timestamp1').addEventListener('input', validateTimestamps);
    document.getElementById('timestamp2').addEventListener('input', validateTimestamps);
    document.getElementById('timestamp3').addEventListener('input', validateTimestamps);
}

function validateTimestamps() {
    const t1 = document.getElementById('timestamp1');
    const t2 = document.getElementById('timestamp2');
    const t3 = document.getElementById('timestamp3');
    const analyzeBtn = document.getElementById('analyzeBtn');

    // Check all fields are valid and filled
    const isValid = t1.checkValidity() && 
                    t2.checkValidity() && 
                    t3.checkValidity() &&
                    t1.value !== '' &&
                    t2.value !== '' &&
                    t3.value !== '';

    analyzeBtn.disabled = !isValid;
}

function showScreen(screenId) {
    document.querySelectorAll('.card').forEach(el => el.classList.add('hidden'));
    const targetScreen = document.getElementById(screenId);
    if (targetScreen) {
        targetScreen.classList.remove('hidden');
        window.scrollTo(0, 0);
    } else {
        console.error(`Screen with ID ${screenId} not found`);
    }
}

async function handleNewAnalysis() {
    try {
        // abort any in-flight work and clear the form & progress
        await cancelAnalysis();
        resetApp();
        // re-enable & relabel the button so it shows "Analyze"
        resetAnalyzeButton();
        // go back to the upload/timestamp screen
        showScreen('initialScreen');
    } catch (error) {
        showError(`Failed to start new analysis: ${error.message}`);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) handleFile(file);
}

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.add('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
}

function handleFile(file) {
    if (!file || !file.type.startsWith('video/')) {
        showError('Please upload a valid video file (MP4, MOV, or AVI)');
        return;
    }
    
    currentFile = file;
    isSharePointFile = false;
    
    const preview = document.getElementById('videoPreview');
    const analyzeBtn = document.getElementById('analyzeBtn');

    // Clean up previous video URL if exists
    if (preview.src) URL.revokeObjectURL(preview.src);
    
    preview.src = URL.createObjectURL(file);
    preview.classList.remove('hidden');
    analyzeBtn.disabled = false;

    document.getElementById('timestamp1').value = '';
    document.getElementById('timestamp2').value = '';
    document.getElementById('timestamp3').value = '';
    validateTimestamps();
}

function showUploadScreen(type) {
    if (type === 'sharepoint') {
        showScreen('sharepointCredScreen');
    } else {
        showScreen('localUploadScreen');
    }
}

async function handleSpCredSubmit(event) {
    event.preventDefault();
    
    const submitBtn = event.target.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Connecting...';

    spCredentials = {
        siteUrl: document.getElementById('spSiteUrl').value.trim(),
        clientId: document.getElementById('spClientId').value.trim(),
        clientSecret: document.getElementById('spClientSecret').value.trim(),
        docLibrary: document.getElementById('spDocLibrary').value.trim()
    };

    try {
        const response = await fetch(`${API_BASE_URL}/api/sharepoint/files`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({
                ...spCredentials,
                doc_library: spCredentials.docLibrary
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || response.statusText);
        }
        
        const files = await response.json();
        renderSpFileList(files);
        showScreen('sharepointFileScreen');
    } catch (error) {
        showError(`SharePoint connection failed: ${error.message}`);
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalText;
    }
}

function renderSpFileList(files) {
    const fileList = document.getElementById('spFileList');
    if (!fileList) return;

    fileList.innerHTML = files.map(file => `
        <div class="sp-file-item">
            <span>${file.name}</span>
            <button class="btn" onclick="handleSpFile('${file.id}')">
                <i class="fas fa-play"></i> Select
            </button>
        </div>
    `).join('');
}

// Add validation function
function validateTimeOrder(t1, t2, t3) {
    const toSeconds = t => {
        const [h, m, s] = t.split(':').map(Number);
        return h * 3600 + m * 60 + s;
    };

    return toSeconds(t1) < toSeconds(t2) && 
           toSeconds(t2) < toSeconds(t3);
}

async function handleSpFile(fileId) {
    const selectBtn = event.target;
    const originalText = selectBtn.innerHTML;
    selectBtn.disabled = true;
    selectBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';

    try {
        const formData = new URLSearchParams({
            ...spCredentials,
            file_id: fileId
        });

        const response = await fetch(`${API_BASE_URL}/api/sharepoint/download`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || response.statusText);
        }
        
        currentFile = await response.blob();
        isSharePointFile = true;
        await startAnalysis();
    } catch (error) {
        showError(`File download failed: ${error.message}`);
    } finally {
        selectBtn.disabled = false;
        selectBtn.innerHTML = originalText;
    }
}

async function startAnalysis() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    analyzeBtn.disabled = true;
    analyzeBtn.onclick = null;
    analyzeBtn.innerText = 'Analyzing…';

    if (!currentFile) {
        showError('Please select a file first!');
        resetAnalyzeButton();
        return;
    }

    const t1 = document.getElementById('timestamp1').value;
    const t2 = document.getElementById('timestamp2').value;
    const t3 = document.getElementById('timestamp3').value;
    if (!validateTimeOrder(t1, t2, t3)) {
      showError('Timestamps must be in ascending order');
      resetAnalyzeButton();
      return;
    }

    showScreen('progressScreen');
    analysisAbortController = new AbortController();

    try {
        const formData = new FormData();
        formData.append('video', currentFile);
        formData.append('timestamp1', t1);
        formData.append('timestamp2', t2);
        formData.append('timestamp3', t3);

        const response = await fetch(`${API_BASE_URL}/api/process-video`, {
            method: 'POST',
            body: formData,
            signal: analysisAbortController.signal
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || response.statusText);
        }

        const { process_id } = await response.json();
        setupProgressTracker(process_id);

    } catch (error) {
        if (error.name !== 'AbortError') {
            showError(`Analysis failed: ${error.message}`);
            showScreen('initialScreen');
        }
    }
}

function setupProgressTracker(processId) {
    if (progressSource) progressSource.close();

    progressSource = new EventSource(`${API_BASE_URL}/api/progress/${processId}`);
    
    progressSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            
            if (data.status === 'completed') {
                handleAnalysisComplete(processId);
                progressSource.close();
            } else if (data.status === 'error') {
                showError(data.error || 'Analysis failed');
                progressSource.close();
                showScreen('initialScreen');
            } else {
                updateProgressUI(data);
            }
        } catch (error) {
            console.error('Error parsing progress:', error);
        }
    };

    progressSource.onerror = () => {
        console.log('SSE error - attempting reconnect');
        setTimeout(() => setupProgressTracker(processId), 2000);
    };
}

async function handleAnalysisComplete(processId) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/results/${processId}`);
        const blob = await response.blob();
        setupDownload(blob);
        showScreen('resultsScreen');
    } catch (error) {
        showError('Failed to retrieve results');
    }
}

function resetAnalyzeButton() {
    const btn = document.getElementById('analyzeBtn');
    btn.disabled = false;
    btn.innerText = 'Start Analysis';
    btn.onclick = startAnalysis;
  }
  

function updateProgressUI(progress) {
    const progressBar = document.getElementById('progressBar');
    const progressMessage = document.getElementById('progressMessage');
    
    progressBar.style.width = `${progress.percent}%`;
    progressMessage.textContent = progress.message;
    
    if (progress.current && progress.total) {
        document.getElementById('frameCounter').textContent = 
            `${progress.current}/${progress.total} seconds processed`;
    }
}

function setupDownload(blob) {
    const url = URL.createObjectURL(blob);
    const downloadBtn = document.getElementById('downloadBtn');
    
    // Clean up previous click handler if exists
    downloadBtn.onclick = null;
    
    downloadBtn.onclick = () => {
        const a = document.createElement('a');
        a.href = url;
        a.download = `stranger_danger_analysis_${new Date().toISOString().slice(0,10)}.csv`;
        document.body.appendChild(a);
        a.click();
        
        // Clean up
        setTimeout(() => {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 100);
    };
}


async function cancelAnalysis() {
    try {
        if (!analysisAbortController) return;

        // Update UI immediately
        const progressMessage = document.getElementById('progressMessage');
        if (progressMessage) {
            progressMessage.textContent = "Cancelling analysis...";
        }

        // Abort ongoing requests
        analysisAbortController.abort();
        
        // Notify backend
        await fetch(`${API_BASE_URL}/api/cancel-analysis`, { 
            method: 'POST'
        });
        
    } catch (error) {
        console.error('Cancellation error:', error);
        throw error;
    } finally {
        analysisAbortController = null;
    }
}

function resetApp() {
    // Clear media resources
    const preview = document.getElementById('videoPreview');
    if (preview.src) URL.revokeObjectURL(preview.src);
    preview.src = '';
    preview.classList.add('hidden');

    // Reset form elements
    document.getElementById('videoInput').value = '';
    
    // Clear progress indicators
    const progressBar = document.getElementById('progressBar');
    if (progressBar) progressBar.style.width = '0%';
    
    const progressMessage = document.getElementById('progressMessage');
    if (progressMessage) progressMessage.textContent = '';
    
    // Clear SharePoint credentials
    const spForm = document.getElementById('spCredForm');
    if (spForm) spForm.reset();
    
    // Force cleanup
    if (progressSource) {
        progressSource.close();
        progressSource = null;
    }
    
    // Reset state
    currentFile = null;
    isSharePointFile = false;
    spCredentials = {};
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `
        <i class="fas fa-exclamation-circle"></i>
        <span>${message}</span>
    `;
    
    document.body.prepend(errorDiv);
    
    setTimeout(() => {
        errorDiv.classList.add('fade-out');
        setTimeout(() => errorDiv.remove(), 500);
    }, 5000);
}


// Make functions available globally for HTML onclick attributes
window.showUploadScreen = showUploadScreen;
window.handleSpCredSubmit = handleSpCredSubmit;
window.handleSpFile = handleSpFile;
window.startAnalysis = startAnalysis;
window.cancelAnalysis = cancelAnalysis;
window.resetApp = resetApp;