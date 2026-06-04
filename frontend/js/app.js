// ── 인증 체크 ────────────────────────────────────────────────
(function checkAuth() {
  if (!localStorage.getItem('access_token')) {
    window.location.href = '/login';
  }
  const username = localStorage.getItem('username');
  if (username) {
    const badge = document.getElementById('username-badge');
    if (badge) badge.textContent = username + ' 님';
  }
})();

function logout() {
  localStorage.removeItem('access_token');
  localStorage.removeItem('username');
  window.location.href = '/login';
}

// ── 초기화 ───────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  await loadVoices();
  pollStatus();
});

// ── 목소리 목록 ──────────────────────────────────────────────
async function loadVoices() {
  const voices = await fetch('/api/voices').then(r => r.json());
  const sel = document.getElementById('voice-select');
  voices.forEach(({ id, name }) => {
    const opt = document.createElement('option');
    opt.value = id;
    opt.textContent = name;
    sel.appendChild(opt);
  });
}

// ── 모델 상태 폴링 (2초) ─────────────────────────────────────
async function pollStatus() {
  try {
    const data = await fetch('/api/status').then(r => r.json());
    applyModelState(data);
  } catch (_) {}
  setTimeout(pollStatus, 2000);
}

function applyModelState({ ready, status, error, loading }) {
  const statusEl = document.getElementById('model-status-text');
  const loadBtn  = document.getElementById('load-model-btn');
  const genBtn   = document.getElementById('generate-btn');
  const initBtn  = document.getElementById('init-avatar-btn');

  statusEl.textContent = status;

  if (ready) {
    statusEl.className = 'ready';
    loadBtn.style.display = 'none';
    genBtn.disabled  = false;
    initBtn.disabled = false;
  } else if (error) {
    statusEl.className = 'error';
    loadBtn.disabled   = false;
    loadBtn.textContent = '재시도';
  } else if (loading) {
    statusEl.className  = '';
    loadBtn.disabled    = true;
    loadBtn.textContent = '로딩 중...';
  }
}

// ── 모델 로드 ────────────────────────────────────────────────
async function loadModel() {
  const loadBtn = document.getElementById('load-model-btn');
  loadBtn.disabled    = true;
  loadBtn.textContent = '로딩 중...';

  await fetch('/api/load_model', { method: 'POST' });

  // SSE로 진행상황 수신
  const es = new EventSource('/api/load_model/stream');
  es.onmessage = ({ data }) => {
    const msg = JSON.parse(data);
    document.getElementById('model-status-text').textContent = msg.status;
    if (msg.ready || msg.error) {
      es.close();
      applyModelState(msg);
    }
  };
  es.onerror = () => es.close();
}

// ── 영상 생성 ────────────────────────────────────────────────
async function generate() {
  const text  = document.getElementById('text-input').value.trim();
  if (!text) return;

  const genBtn   = document.getElementById('generate-btn');
  const statusEl = document.getElementById('gen-status');
  const videoEl  = document.getElementById('video-output');
  const placeholder = document.getElementById('video-placeholder');

  genBtn.disabled    = true;
  statusEl.textContent = '요청 중...';

  const form = new FormData();
  form.append('text',  text);
  form.append('voice', document.getElementById('voice-select').value);

  await readSSE('/api/generate', form, ({ status, error, done, video_path }) => {
    if (status)  statusEl.textContent = status;
    if (error)   statusEl.textContent = `오류: ${error}`;
    if (video_path) {
      videoEl.src = `/api/video?path=${encodeURIComponent(video_path)}`;
      videoEl.style.display     = 'block';
      placeholder.style.display = 'none';
      videoEl.play();
    }
  });

  genBtn.disabled = false;
}

// ── 아바타 초기화 ────────────────────────────────────────────
async function initAvatar() {
  const fileInput = document.getElementById('avatar-file');
  if (!fileInput.files.length) { alert('사진을 선택해주세요.'); return; }

  const btn      = document.getElementById('init-avatar-btn');
  const statusEl = document.getElementById('avatar-status');
  const preview  = document.getElementById('lp-preview');

  btn.disabled = true;

  const form = new FormData();
  form.append('file',          fileInput.files[0]);
  form.append('driving_style', document.getElementById('driving-style').value);
  form.append('motion',        document.getElementById('motion').value);
  form.append('region',        document.getElementById('region').value);
  form.append('bbox_shift',    document.getElementById('bbox-shift').value);

  await readSSE('/api/init_avatar', form, ({ status, error, preview_path }) => {
    if (status)       statusEl.textContent = status;
    if (error)        statusEl.textContent = `오류: ${error}`;
    if (preview_path) {
      preview.src    = `/api/video?path=${encodeURIComponent(preview_path)}`;
      preview.hidden = false;
    }
  });

  btn.disabled = false;
}

// ── SSE 유틸 (fetch stream) ──────────────────────────────────
async function readSSE(url, formData, onMessage) {
  const res    = await fetch(url, { method: 'POST', body: formData });
  const reader = res.body.getReader();
  const dec    = new TextDecoder();
  let   buf    = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split('\n');
    buf = lines.pop();
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const msg = JSON.parse(line.slice(6));
      onMessage(msg);
      if (msg.done || msg.error) return;
    }
  }
}
