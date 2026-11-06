/**
 * Handwritten Digit Recognition - Frontend JavaScript
 * Handles drawing canvas, image upload, and prediction display
 */

// ============================================================
// GLOBAL STATE
// ============================================================
const state = {
    isDrawing: false,
    lastX: 0,
    lastY: 0,
    brushSize: 20,
    brushColor: '#000000',
    canvas: null,
    ctx: null,
    predictionMade: false
};

// ============================================================
// INITIALIZATION
// ============================================================
document.addEventListener('DOMContentLoaded', function() {
    initializeCanvas();
    initializeUpload();
    initializeSmoothScroll();
    initializeNavbar();
    initializePreviewCanvas();
});

function initializeCanvas() {
    const canvas = document.getElementById('drawingCanvas');
    if (!canvas) return;
    
    state.canvas = canvas;
    state.ctx = canvas.getContext('2d');
    
    // Set canvas resolution for crisp drawing
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    state.ctx.scale(dpr, dpr);
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    
    // Initialize canvas with white background
    state.ctx.fillStyle = '#ffffff';
    state.ctx.fillRect(0, 0, rect.width, rect.height);
    
    // Mouse events
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    // Touch events
    canvas.addEventListener('touchstart', handleTouchStart, { passive: false });
    canvas.addEventListener('touchmove', handleTouchMove, { passive: false });
    canvas.addEventListener('touchend', stopDrawing);
    
    // Brush size control
    const brushSizeInput = document.getElementById('brushSize');
    const brushSizeValue = document.getElementById('brushSizeValue');
    if (brushSizeInput && brushSizeValue) {
        brushSizeInput.addEventListener('input', function() {
            state.brushSize = parseInt(this.value);
            brushSizeValue.textContent = state.brushSize;
        });
    }
    
    // Prevent context menu on canvas
    canvas.addEventListener('contextmenu', e => e.preventDefault());
}

function initializeUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    if (!uploadArea || !fileInput) return;
    
    // Click to upload
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('dragover');
        
        const file = e.dataTransfer.files[0];
        if (file) handleFileUpload(file);
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleFileUpload(file);
    });
}

function initializeSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const target = document.querySelector(targetId);
            if (target) {
                e.preventDefault();
                const navHeight = document.getElementById('mainNav').offsetHeight;
                const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - navHeight;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
}

function initializeNavbar() {
    const nav = document.getElementById('mainNav');
    if (!nav) return;
    
    window.addEventListener('scroll', function() {
        if (window.scrollY > 50) {
            nav.classList.add('scrolled');
        } else {
            nav.classList.remove('scrolled');
        }
    });
}

function initializePreviewCanvas() {
    const canvas = document.getElementById('previewCanvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    
    // Draw a sample "7" for demo
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, rect.width, rect.height);
    
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 8;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    ctx.beginPath();
    ctx.moveTo(40, 40);
    ctx.lineTo(160, 40);
    ctx.lineTo(60, 160);
    ctx.stroke();
}

// ============================================================
// DRAWING FUNCTIONS
// ============================================================
function getCanvasCoordinates(e) {
    const rect = state.canvas.getBoundingClientRect();
    const clientX = e.clientX || (e.touches && e.touches[0].clientX);
    const clientY = e.clientY || (e.touches && e.touches[0].clientY);
    
    return {
        x: clientX - rect.left,
        y: clientY - rect.top
    };
}

function startDrawing(e) {
    if (e.button !== undefined && e.button !== 0) return; // Only left mouse button
    
    state.isDrawing = true;
    const coords = getCanvasCoordinates(e);
    state.lastX = coords.x;
    state.lastY = coords.y;
    
    // Draw initial dot
    drawDot(coords.x, coords.y);
}

function draw(e) {
    if (!state.isDrawing) return;
    e.preventDefault();
    
    const coords = getCanvasCoordinates(e);
    drawLine(state.lastX, state.lastY, coords.x, coords.y);
    
    state.lastX = coords.x;
    state.lastY = coords.y;
}

function drawLine(x1, y1, x2, y2) {
    const ctx = state.ctx;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = state.brushColor;
    ctx.lineWidth = state.brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.stroke();
}

function drawDot(x, y) {
    const ctx = state.ctx;
    ctx.beginPath();
    ctx.arc(x, y, state.brushSize / 2, 0, Math.PI * 2);
    ctx.fillStyle = state.brushColor;
    ctx.fill();
}

function stopDrawing() {
    state.isDrawing = false;
}

function handleTouchStart(e) {
    e.preventDefault();
    if (e.touches.length === 1) {
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent('mousedown', {
            clientX: touch.clientX,
            clientY: touch.clientY,
            button: 0
        });
        state.canvas.dispatchEvent(mouseEvent);
    }
}

function handleTouchMove(e) {
    e.preventDefault();
    if (e.touches.length === 1 && state.isDrawing) {
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent('mousemove', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        state.canvas.dispatchEvent(mouseEvent);
    }
}

function clearCanvas() {
    if (!state.ctx || !state.canvas) return;
    
    const rect = state.canvas.getBoundingClientRect();
    state.ctx.fillStyle = '#ffffff';
    state.ctx.fillRect(0, 0, rect.width, rect.height);
    
    // Reset prediction display
    hideResults();
}

function canvasToBase64() {
    if (!state.canvas) return null;
    
    // Create a temporary canvas for processing
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    
    // Use 28x28 for model input (but we'll send larger for better quality)
    const size = 200;
    tempCanvas.width = size;
    tempCanvas.height = size;
    
    // Fill with black (MNIST background)
    tempCtx.fillStyle = '#000000';
    tempCtx.fillRect(0, 0, size, size);
    
    // Draw the user's drawing in white (MNIST foreground)
    tempCtx.drawImage(state.canvas, 0, 0, size, size);
    
    // Convert to base64
    return tempCanvas.toDataURL('image/png');
}

// ============================================================
// IMAGE UPLOAD FUNCTIONS
// ============================================================
function handleFileUpload(file) {
    // Validate file type
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp'];
    if (!allowedTypes.includes(file.type)) {
        showError('Invalid file type. Please upload PNG, JPG, JPEG, GIF, or BMP.');
        return;
    }
    
    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('File too large. Maximum size is 16 MB.');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = function(e) {
        showImagePreview(e.target.result);
    };
    reader.readAsDataURL(file);
}

function showImagePreview(dataUrl) {
    const uploadArea = document.getElementById('uploadArea');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    
    if (uploadArea) uploadArea.style.display = 'none';
    if (imagePreview) imagePreview.classList.remove('d-none');
    if (previewImg) previewImg.src = dataUrl;
}

function clearUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const imagePreview = document.getElementById('imagePreview');
    const fileInput = document.getElementById('fileInput');
    const previewImg = document.getElementById('previewImg');
    
    if (uploadArea) uploadArea.style.display = 'block';
    if (imagePreview) imagePreview.classList.add('d-none');
    if (previewImg) previewImg.src = '';
    if (fileInput) fileInput.value = '';
    
    hideResults();
}

// ============================================================
// PREDICTION FUNCTIONS
// ============================================================
async function predictDrawing() {
    const base64Image = canvasToBase64();
    if (!base64Image) {
        showError('Please draw a digit first.');
        return;
    }
    
    // Check if canvas is mostly empty (white)
    if (isCanvasEmpty()) {
        showError('Canvas appears empty. Please draw a digit.');
        return;
    }
    
    await makePrediction(base64Image, 'drawing');
}

async function predictUpload() {
    const previewImg = document.getElementById('previewImg');
    if (!previewImg || !previewImg.src) {
        showError('Please upload an image first.');
        return;
    }
    
    // Convert preview image to base64
    const base64Image = previewImg.src;
    await makePrediction(base64Image, 'upload');
}

async function makePrediction(base64Image, source) {
    const predictBtn = document.getElementById('predictBtn');
    const uploadPredictBtn = document.querySelector('#imagePreview button.btn-primary');
    const btn = source === 'drawing' ? predictBtn : uploadPredictBtn;
    
    // Show loading state
    setButtonLoading(btn, true);
    hideError();
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: base64Image })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Prediction failed');
        }
        
        if (data.success) {
            showResults(data);
            state.predictionMade = true;
        } else {
            throw new Error(data.error || 'Prediction failed');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        showError(error.message || 'Failed to get prediction. Please try again.');
    } finally {
        setButtonLoading(btn, false);
    }
}

function setButtonLoading(button, loading) {
    if (!button) return;
    
    const btnText = button.querySelector('.btn-text');
    const spinner = button.querySelector('.spinner-border');
    
    if (loading) {
        button.disabled = true;
        if (btnText) btnText.textContent = 'Predicting...';
        if (spinner) spinner.classList.remove('d-none');
    } else {
        button.disabled = false;
        if (btnText) btnText.textContent = button.id === 'predictBtn' ? 'Predict Digit' : 'Predict';
        if (spinner) spinner.classList.add('d-none');
    }
}

// ============================================================
// RESULTS DISPLAY
// ============================================================
function showResults(data) {
    const resultsSection = document.getElementById('resultsSection');
    const predictedDigit = document.getElementById('predictedDigit');
    const confidenceValue = document.getElementById('confidenceValue');
    const predictionCircle = document.getElementById('predictionCircle');
    const probabilityBars = document.getElementById('probabilityBars');
    
    // Show results section with animation
    if (resultsSection) {
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        resultsSection.classList.add('animate-fade-in-up');
    }
    
    // Update main prediction
    if (predictedDigit) {
        predictedDigit.textContent = data.predicted_digit;
        // Animate number change
        predictedDigit.style.transform = 'scale(1.2)';
        setTimeout(() => predictedDigit.style.transform = 'scale(1)', 150);
    }
    
    if (confidenceValue) {
        confidenceValue.textContent = data.confidence.toFixed(1) + '%';
    }
    
    // Update prediction circle color based on confidence
    if (predictionCircle) {
        const confidence = data.confidence;
        if (confidence >= 90) {
            predictionCircle.style.background = 'linear-gradient(135deg, #10b981, #34d399)';
        } else if (confidence >= 70) {
            predictionCircle.style.background = 'linear-gradient(135deg, #f59e0b, #fbbf24)';
        } else {
            predictionCircle.style.background = 'linear-gradient(135deg, #ef4444, #f87171)';
        }
    }
    
    // Build probability bars
    if (probabilityBars && data.probabilities) {
        probabilityBars.innerHTML = '';
        
        // Find highest probability
        const probs = Object.entries(data.probabilities).map(([digit, prob]) => ({
            digit: parseInt(digit),
            prob: prob
        })).sort((a, b) => b.prob - a.prob);
        
        const highestDigit = probs[0].digit;
        
        probs.forEach(({ digit, prob }) => {
            const bar = document.createElement('div');
            bar.className = 'probability-bar';
            bar.style.animationDelay = `${digit * 50}ms`;
            
            const isHighest = digit === highestDigit;
            
            bar.innerHTML = `
                <div class="probability-bar-header">
                    <span class="probability-label">${digit}</span>
                    <span class="probability-value">${prob.toFixed(1)}%</span>
                </div>
                <div class="probability-bar-track">
                    <div class="probability-bar-fill ${isHighest ? 'highest' : ''}" 
                         style="width: 0%" 
                         data-width="${prob}"></div>
                </div>
            `;
            
            probabilityBars.appendChild(bar);
        });
        
        // Animate bars
        setTimeout(() => {
            document.querySelectorAll('.probability-bar-fill').forEach(fill => {
                const width = fill.getAttribute('data-width');
                fill.style.width = width + '%';
            });
        }, 100);
    }
}

function hideResults() {
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
        resultsSection.style.display = 'none';
        resultsSection.classList.remove('animate-fade-in-up');
    }
    state.predictionMade = false;
}

function resetPrediction() {
    hideResults();
    clearCanvas();
    clearUpload();
    
    // Scroll back to drawing section
    const drawSection = document.getElementById('draw');
    if (drawSection) {
        drawSection.scrollIntoView({ behavior: 'smooth' });
    }
}

// ============================================================
// UTILITY FUNCTIONS
// ============================================================
function isCanvasEmpty() {
    if (!state.canvas) return true;
    
    const ctx = state.canvas.getContext('2d');
    const rect = state.canvas.getBoundingClientRect();
    const imageData = ctx.getImageData(0, 0, rect.width, rect.height);
    const data = imageData.data;
    
    // Count non-white pixels (allowing for anti-aliasing)
    let nonWhitePixels = 0;
    const threshold = 250; // Pixels with any channel < 250 are considered "drawn"
    
    for (let i = 0; i < data.length; i += 4) {
        if (data[i] < threshold || data[i+1] < threshold || data[i+2] < threshold) {
            nonWhitePixels++;
        }
    }
    
    // Consider empty if less than 0.5% of pixels are non-white
    const totalPixels = rect.width * rect.height;
    return (nonWhitePixels / totalPixels) < 0.005;
}

function showError(message) {
    // Create or update error toast
    let toast = document.getElementById('errorToast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'errorToast';
        toast.className = 'toast align-items-center text-white bg-danger border-0 position-fixed bottom-0 end-0 m-3';
        toast.setAttribute('role', 'alert');
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body" id="toastMessage"></div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        document.body.appendChild(toast);
    }
    
    const toastMessage = document.getElementById('toastMessage');
    if (toastMessage) toastMessage.textContent = message;
    
    const bsToast = new bootstrap.Toast(toast, { delay: 4000 });
    bsToast.show();
}

function hideError() {
    const toast = document.getElementById('errorToast');
    if (toast) {
        const bsToast = bootstrap.Toast.getInstance(toast);
        if (bsToast) bsToast.hide();
    }
}

// ============================================================
// KEYBOARD SHORTCUTS
// ============================================================
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to predict
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if (!state.predictionMade) {
            predictDrawing();
        }
    }
    
    // Escape to clear
    if (e.key === 'Escape') {
        clearCanvas();
        clearUpload();
        hideResults();
    }
    
    // 'c' to clear canvas
    if (e.key === 'c' || e.key === 'C') {
        if (document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
            clearCanvas();
        }
    }
});

// ============================================================
// RESIZE HANDLER
// ============================================================
let resizeTimeout;
window.addEventListener('resize', function() {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(function() {
        // Reinitialize canvas on resize
        initializeCanvas();
    }, 250);
});

// ============================================================
// EXPORT FOR TESTING
// ============================================================
window.DigitRecognizer = {
    state,
    clearCanvas,
    predictDrawing,
    predictUpload,
    canvasToBase64
};