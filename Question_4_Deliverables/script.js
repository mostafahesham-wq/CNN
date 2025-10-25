// script.js - frontend logic for CIFAR-10 FastAPI

// API endpoint - relative path to work when served from FastAPI static files
// If running separately, change to full URL like 'http://127.0.0.1:8000/predict'
const API_PREDICT = '/predict';

const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');
const previewArea = document.getElementById('previewArea');
const previewCanvas = document.getElementById('previewCanvas');
const imageInfo = document.getElementById('imageInfo');
const predictBtn = document.getElementById('predictBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const predList = document.getElementById('predList');
const barsCanvas = document.getElementById('barsCanvas');
const errorMsg = document.getElementById('errorMsg');
const historyList = document.getElementById('historyList');
const clearBtn = document.getElementById('clearBtn');
const sampleBtn = document.getElementById('sampleBtn');
const captureBtn = document.getElementById('captureBtn');
const webcamModal = document.getElementById('webcamModal');
const webcamVideo = document.getElementById('webcamVideo');
const snapBtn = document.getElementById('snapBtn');
const closeCamBtn = document.getElementById('closeCamBtn');
const darkToggle = document.getElementById('darkToggle');

let currentBlob = null;
let mediaStream = null;

// helpers
function show(el){el.classList.remove('hidden')}
function hide(el){el.classList.add('hidden')}
function setError(msg){errorMsg.textContent = msg; show(errorMsg)}
function clearError(){errorMsg.textContent=''; hide(errorMsg)}

// preview drawing (maintain aspect ratio into 128x128 canvas)
function drawPreviewFromImage(img){
  const ctx = previewCanvas.getContext('2d');
  const SIZE = 128;
  previewCanvas.width = SIZE; previewCanvas.height = SIZE;
  ctx.fillStyle = '#fff';
  ctx.fillRect(0,0,SIZE,SIZE);
  // compute fit
  const ratio = Math.min(SIZE / img.width, SIZE / img.height);
  const nw = img.width * ratio; const nh = img.height * ratio;
  const dx = (SIZE - nw)/2; const dy = (SIZE - nh)/2;
  ctx.drawImage(img, dx, dy, nw, nh);
}

function setCurrentBlob(blob, filename){
  currentBlob = blob;
  const img = new Image();
  img.onload = ()=>{
    drawPreviewFromImage(img);
    show(previewArea);
    imageInfo.textContent = filename || 'Uploaded image';
  };
  img.src = URL.createObjectURL(blob);
}

// drag & drop
['dragenter','dragover'].forEach(ev => {
  dropZone.addEventListener(ev, e=>{e.preventDefault(); dropZone.classList.add('dragover')});
});
['dragleave','drop'].forEach(ev => {
  dropZone.addEventListener(ev, e=>{e.preventDefault(); dropZone.classList.remove('dragover')});
});

dropZone.addEventListener('drop', e=>{
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if(f) handleFile(f);
});

fileInput.addEventListener('change', e=>{
  const f = e.target.files[0]; if(f) handleFile(f);
});

function handleFile(file){
  if(!file.type.startsWith('image/')){setError('Please upload an image.'); return}
  clearError();
  setCurrentBlob(file, file.name);
}

// sample image - creates a simple colored test image
sampleBtn.addEventListener('click', ()=>{
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');
  
  // Create a simple gradient pattern as sample
  const gradient = ctx.createLinearGradient(0, 0, 128, 128);
  gradient.addColorStop(0, '#3b82f6');
  gradient.addColorStop(1, '#8b5cf6');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 128, 128);
  
  // Add some shapes to make it look like something
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  ctx.arc(64, 64, 30, 0, Math.PI * 2);
  ctx.fill();
  
  canvas.toBlob(blob => {
    if(blob) setCurrentBlob(blob, 'sample.png');
  });
});

// clear
clearBtn.addEventListener('click', ()=>{
  currentBlob = null; 
  hide(previewArea); 
  hide(results); 
  clearError(); 
  hide(loading);
});

// predict
predictBtn.addEventListener('click', async ()=>{
  if(!currentBlob){setError('No image selected'); return}
  clearError(); hide(results); show(loading);

  const form = new FormData();
  form.append('file', currentBlob, 'image.png');

  try{
    const resp = await fetch(API_PREDICT, {method:'POST', body: form});
    if(!resp.ok){
      const txt = await resp.text(); 
      throw new Error(`Server error: ${resp.status} - ${txt}`);
    }
    const data = await resp.json();
    // expected format: {predictions: [{class: 'airplane', prob: 0.7}, ...]}
    renderResults(data);
  }catch(err){
    setError(err.message || 'Prediction failed');
    console.error('Prediction error:', err);
  }finally{
    hide(loading);
  }
});

function renderResults(data){
  if(!data || !Array.isArray(data.predictions)){
    setError('Invalid response from server'); return;
  }
  show(results); clearError(); predList.innerHTML='';
  const preds = data.predictions.slice(0,5);
  preds.forEach(p => {
    const li = document.createElement('li'); li.className='pred-item';
    const left = document.createElement('div');
    left.innerHTML = `<strong>${p.class}</strong><small>${(p.prob*100).toFixed(2)}%</small>`;
    const right = document.createElement('div');
    right.innerHTML = `<small>${p.prob.toFixed(3)}</small>`;
    li.appendChild(left); li.appendChild(right); predList.appendChild(li);
  });

  drawBars(preds.slice(0,3));
  pushHistory(preds[0]);
}

function drawBars(top3){
  const ctx = barsCanvas.getContext('2d');
  ctx.clearRect(0,0,barsCanvas.width,barsCanvas.height);
  const padding = 10; const barH = 28; const gap = 14;
  top3.forEach((p,i)=>{
    const x = padding; const y = padding + i*(barH+gap);
    const label = `${p.class} ${(p.prob*100).toFixed(1)}%`;
    ctx.font = '14px Arial'; ctx.fillStyle = '#222'; ctx.fillText(label, x, y+16);
    const barX = 160; const barW = barsCanvas.width - barX - padding;
    // background
    ctx.fillStyle = '#eee'; ctx.fillRect(barX, y, barW, barH);
    // value
    ctx.fillStyle = '#2563eb'; ctx.fillRect(barX, y, Math.max(2, barW * p.prob), barH);
  });
}

// history functions
function loadHistory(){
  try {
    const raw = localStorage.getItem('cifar_history');
    return raw ? JSON.parse(raw) : [];
  } catch(e) {
    console.error('Error loading history:', e);
    return [];
  }
}

function saveHistory(arr){ 
  try {
    localStorage.setItem('cifar_history', JSON.stringify(arr));
  } catch(e) {
    console.error('Error saving history:', e);
  }
}

function pushHistory(top){
  const h = loadHistory();
  const entry = {time: Date.now(), class: top.class, prob: top.prob};
  h.unshift(entry); 
  if(h.length>10) h.pop(); 
  saveHistory(h); 
  renderHistory();
}

function renderHistory(){
  const h = loadHistory(); 
  historyList.innerHTML='';
  h.forEach(it=>{
    const li = document.createElement('li'); 
    li.className='history-item';
    const d = new Date(it.time); 
    li.textContent = `${d.toLocaleString()} — ${it.class} (${(it.prob*100).toFixed(1)}%)`;
    historyList.appendChild(li);
  });
}

// initialize history on load
renderHistory();

// webcam functionality
captureBtn.addEventListener('click', async ()=>{
  hide(errorMsg); 
  webcamModal.classList.remove('hidden');
  try{
    mediaStream = await navigator.mediaDevices.getUserMedia({video:true});
    webcamVideo.srcObject = mediaStream;
  }catch(e){ 
    setError('Unable to access webcam'); 
    webcamModal.classList.add('hidden');
    console.error('Webcam error:', e);
  }
});

closeCamBtn.addEventListener('click', ()=>{ 
  stopWebcam(); 
  webcamModal.classList.add('hidden');
});

snapBtn.addEventListener('click', ()=>{
  const c = document.createElement('canvas'); 
  c.width=128; 
  c.height=128;
  const ctx = c.getContext('2d'); 
  ctx.drawImage(webcamVideo, 0, 0, c.width, c.height);
  c.toBlob(b=> {
    if(b) setCurrentBlob(b, 'webcam.png');
  });
  stopWebcam(); 
  webcamModal.classList.add('hidden');
});

function stopWebcam(){ 
  if(mediaStream){ 
    mediaStream.getTracks().forEach(t=>t.stop()); 
    mediaStream=null; 
    webcamVideo.srcObject=null;
  }
}

// dark mode toggle
darkToggle.addEventListener('change', ()=>{
  document.body.classList.toggle('dark', darkToggle.checked);
  // Save preference
  try {
    localStorage.setItem('darkMode', darkToggle.checked);
  } catch(e) {
    console.error('Error saving dark mode preference:', e);
  }
});

// Load dark mode preference on startup
try {
  const darkMode = localStorage.getItem('darkMode');
  if(darkMode === 'true') {
    darkToggle.checked = true;
    document.body.classList.add('dark');
  }
} catch(e) {
  console.error('Error loading dark mode preference:', e);
}

// Click on drop zone to open file dialog
dropZone.addEventListener('click', ()=> fileInput.click());