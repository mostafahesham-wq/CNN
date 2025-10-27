// API Configuration
const API_URL = window.location.origin;

// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        const tabName = tab.dataset.tab;
       
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
       
        tab.classList.add('active');
        document.getElementById(tabName).classList.add('active');
    });
});

// Single Image Upload
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewArea = document.getElementById('previewArea');
const previewImage = document.getElementById('previewImage');
const classifyBtn = document.getElementById('classifyBtn');
const clearBtn = document.getElementById('clearBtn');
const loading = document.getElementById('loading');
const result = document.getElementById('result');
const error = document.getElementById('error');
let selectedFile = null;

uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        previewArea.classList.add('active');
        result.classList.remove('active');
        error.classList.remove('active');
    };
    reader.readAsDataURL(file);
}

clearBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    uploadArea.style.display = 'block';
    previewArea.classList.remove('active');
    result.classList.remove('active');
    error.classList.remove('active');
});

classifyBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    const formData = new FormData();
    formData.append('file', selectedFile);
    loading.classList.add('active');
    result.classList.remove('active');
    error.classList.remove('active');
    classifyBtn.disabled = true;
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });
        if (!response.ok) {
            throw new Error('Classification failed');
        }
        const data = await response.json();
        displayResult(data);
    } catch (err) {
        showError(err.message);
    } finally {
        loading.classList.remove('active');
        classifyBtn.disabled = false;
    }
});

function displayResult(data) {
    let html = '<h3>Classification Result</h3>';
    html += '<div class="prediction">';
    html += `<div class="class-name">Class: ${data.class}</div>`;
    html += `<div class="confidence">Confidence: ${(data.confidence * 100).toFixed(2)}%</div>`;
   
    if (data.probabilities) {
        html += '<div class="probabilities">';
        const sortedProbs = Object.entries(data.probabilities)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5);
       
        sortedProbs.forEach(([cls, prob]) => {
            html += `
                <div class="prob-item">
                    <div class="prob-label">${cls}</div>
                    <div class="prob-bar">
                        <div class="prob-fill" style="width: ${prob * 100}%"></div>
                    </div>
                    <div class="prob-value">${(prob * 100).toFixed(1)}%</div>
                </div>
            `;
        });
        html += '</div>';
    }
   
    html += '</div>';
    result.innerHTML = html;
    result.classList.add('active');
}

function showError(message) {
    error.textContent = `Error: ${message}`;
    error.classList.add('active');
}

// Secure Endpoint
const uploadAreaSecure = document.getElementById('uploadAreaSecure');
const fileInputSecure = document.getElementById('fileInputSecure');
const previewAreaSecure = document.getElementById('previewAreaSecure');
const previewImageSecure = document.getElementById('previewImageSecure');
const classifyBtnSecure = document.getElementById('classifyBtnSecure');
const clearBtnSecure = document.getElementById('clearBtnSecure');
const loadingSecure = document.getElementById('loadingSecure');
const resultSecure = document.getElementById('resultSecure');
const errorSecure = document.getElementById('errorSecure');
const apiKeyInput = document.getElementById('apiKey');
let selectedFileSecure = null;

uploadAreaSecure.addEventListener('click', () => fileInputSecure.click());

uploadAreaSecure.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadAreaSecure.classList.add('dragover');
});

uploadAreaSecure.addEventListener('dragleave', () => {
    uploadAreaSecure.classList.remove('dragover');
});

uploadAreaSecure.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadAreaSecure.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFileSelectSecure(e.dataTransfer.files[0]);
    }
});

fileInputSecure.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFileSelectSecure(e.target.files[0]);
    }
});

function handleFileSelectSecure(file) {
    selectedFileSecure = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImageSecure.src = e.target.result;
        uploadAreaSecure.style.display = 'none';
        previewAreaSecure.classList.add('active');
        resultSecure.classList.remove('active');
        errorSecure.classList.remove('active');
    };
    reader.readAsDataURL(file);
}

clearBtnSecure.addEventListener('click', () => {
    selectedFileSecure = null;
    fileInputSecure.value = '';
    uploadAreaSecure.style.display = 'block';
    previewAreaSecure.classList.remove('active');
    resultSecure.classList.remove('active');
    errorSecure.classList.remove('active');
});

classifyBtnSecure.addEventListener('click', async () => {
    if (!selectedFileSecure) return;
    const apiKey = apiKeyInput.value.trim();
    if (!apiKey) {
        showErrorSecure('Please enter an API key');
        return;
    }
    const formData = new FormData();
    formData.append('file', selectedFileSecure);
    loadingSecure.classList.add('active');
    resultSecure.classList.remove('active');
    errorSecure.classList.remove('active');
    classifyBtnSecure.disabled = true;
    try {
        const response = await fetch(`${API_URL}/predict/secure`, {
            method: 'POST',
            headers: {
                'X-API-KEY': apiKey
            },
            body: formData
        });
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || 'Classification failed');
        }
        const data = await response.json();
        displayResultSecure(data);
    } catch (err) {
        showErrorSecure(err.message);
    } finally {
        loadingSecure.classList.remove('active');
        classifyBtnSecure.disabled = false;
    }
});

function displayResultSecure(data) {
    let html = '<h3>Classification Result (Secure)</h3>';
    html += '<div class="prediction">';
    html += `<div class="class-name">Class: ${data.class}</div>`;
    html += `<div class="confidence">Confidence: ${(data.confidence * 100).toFixed(2)}%</div>`;
   
    if (data.probabilities) {
        html += '<div class="probabilities">';
        const sortedProbs = Object.entries(data.probabilities)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5);
       
        sortedProbs.forEach(([cls, prob]) => {
            html += `
                <div class="prob-item">
                    <div class="prob-label">${cls}</div>
                    <div class="prob-bar">
                        <div class="prob-fill" style="width: ${prob * 100}%"></div>
                    </div>
                    <div class="prob-value">${(prob * 100).toFixed(1)}%</div>
                </div>
            `;
        });
        html += '</div>';
    }
   
    html += '</div>';
    resultSecure.innerHTML = html;
    resultSecure.classList.add('active');
}

function showErrorSecure(message) {
    errorSecure.textContent = `Error: ${message}`;
    errorSecure.classList.add('active');
}

// Batch Upload
const uploadAreaBatch = document.getElementById('uploadAreaBatch');
const fileInputBatch = document.getElementById('fileInputBatch');
const previewAreaBatch = document.getElementById('previewAreaBatch');
const batchPreview = document.getElementById('batchPreview');
const classifyBtnBatch = document.getElementById('classifyBtnBatch');
const clearBtnBatch = document.getElementById('clearBtnBatch');
const loadingBatch = document.getElementById('loadingBatch');
const resultBatch = document.getElementById('resultBatch');
const errorBatch = document.getElementById('errorBatch');
const apiKeyBatch = document.getElementById('apiKeyBatch');
let selectedFilesBatch = [];

uploadAreaBatch.addEventListener('click', () => fileInputBatch.click());

uploadAreaBatch.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadAreaBatch.classList.add('dragover');
});

uploadAreaBatch.addEventListener('dragleave', () => {
    uploadAreaBatch.classList.remove('dragover');
});

uploadAreaBatch.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadAreaBatch.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFileSelectBatch(Array.from(e.dataTransfer.files));
    }
});

fileInputBatch.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFileSelectBatch(Array.from(e.target.files));
    }
});

function handleFileSelectBatch(files) {
    selectedFilesBatch = files;
    batchPreview.innerHTML = '';
   
    files.forEach((file, index) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const div = document.createElement('div');
            div.className = 'batch-image';
            div.innerHTML = `
                <img src="${e.target.result}" alt="${file.name}">
                <button class="remove" onclick="removeBatchImage(${index})">×</button>
            `;
            batchPreview.appendChild(div);
        };
        reader.readAsDataURL(file);
    });
    uploadAreaBatch.style.display = 'none';
    previewAreaBatch.classList.add('active');
    resultBatch.classList.remove('active');
    errorBatch.classList.remove('active');
}

window.removeBatchImage = (index) => {
    selectedFilesBatch.splice(index, 1);
    if (selectedFilesBatch.length === 0) {
        clearBtnBatch.click();
    } else {
        handleFileSelectBatch(selectedFilesBatch);
    }
};

clearBtnBatch.addEventListener('click', () => {
    selectedFilesBatch = [];
    fileInputBatch.value = '';
    batchPreview.innerHTML = '';
    uploadAreaBatch.style.display = 'block';
    previewAreaBatch.classList.remove('active');
    resultBatch.classList.remove('active');
    errorBatch.classList.remove('active');
});

classifyBtnBatch.addEventListener('click', async () => {
    if (selectedFilesBatch.length === 0) return;
    const apiKey = apiKeyBatch.value.trim();
    if (!apiKey) {
        showErrorBatch('Please enter an API key');
        return;
    }
    const formData = new FormData();
    selectedFilesBatch.forEach(file => {
        formData.append('files', file);
    });
    loadingBatch.classList.add('active');
    resultBatch.classList.remove('active');
    errorBatch.classList.remove('active');
    classifyBtnBatch.disabled = true;
    try {
        const response = await fetch(`${API_URL}/predict/batch`, {
            method: 'POST',
            headers: {
                'X-API-KEY': apiKey
            },
            body: formData
        });
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || 'Batch classification failed');
        }
        const data = await response.json();
        displayResultBatch(data);
    } catch (err) {
        showErrorBatch(err.message);
    } finally {
        loadingBatch.classList.remove('active');
        classifyBtnBatch.disabled = false;
    }
});

function displayResultBatch(data) {
    let html = '<h3>Batch Classification Results</h3>';
    html += '<div class="batch-results">';
   
    if (data.results && Array.isArray(data.results)) {
        data.results.forEach((item, index) => {
            const file = selectedFilesBatch[index];
            const reader = new FileReader();
            reader.onload = (e) => {
                const resultItem = document.createElement('div');
                resultItem.className = 'batch-result-item';
                resultItem.innerHTML = `
                    <img src="${e.target.result}" alt="${file.name}" class="batch-result-image">
                    <div class="batch-result-info">
                        <div class="batch-result-name">${file.name}</div>
                        <div class="batch-result-class">${item.class}</div>
                        <div class="batch-result-conf">Confidence: ${(item.confidence * 100).toFixed(2)}%</div>
                    </div>
                `;
                document.querySelector('.batch-results').appendChild(resultItem);
            };
            reader.readAsDataURL(file);
        });
    }
   
    html += '</div>';
    resultBatch.innerHTML = html;
    resultBatch.classList.add('active');
}

function showErrorBatch(message) {
    errorBatch.textContent = `Error: ${message}`;
    errorBatch.classList.add('active');
}