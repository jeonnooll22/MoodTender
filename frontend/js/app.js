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

// Redis 블랙리스트 연동 로그아웃
let isModelReady = false;

async function logout() {
  const token = localStorage.getItem('access_token');
  if (token) {
    try {
      await fetch('/api/logout', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' }
      });
    } catch (e) { console.error("로그아웃 통신 에러:", e); }
  }
  localStorage.removeItem('access_token');
  localStorage.removeItem('username');
  window.location.href = '/login';
}

// ── 초기화 ───────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  const logoutBtn = document.getElementById('logout-btn');
  if (logoutBtn) logoutBtn.addEventListener('click', logout);

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

  isModelReady = Boolean(ready);
  statusEl.textContent = status;

  if (ready) {
    statusEl.className = 'ready';
    loadBtn.style.display = 'none';
    genBtn.disabled  = false;
    initBtn.disabled = false;
  } else if (error) {
    statusEl.className  = 'error';
    loadBtn.disabled    = false;
    genBtn.disabled     = true;
    loadBtn.textContent = '재시도';
  } else if (loading) {
    statusEl.className  = '';
    loadBtn.disabled    = true;
    genBtn.disabled     = true;
    loadBtn.textContent = '로딩 중...';
  }
}

// ── 모델 로드 ────────────────────────────────────────────────
async function loadModel() {
  const loadBtn = document.getElementById('load-model-btn');
  loadBtn.disabled    = true;
  loadBtn.textContent = '로딩 중...';

  await fetch('/api/load_model', { method: 'POST' });

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

// ── 영상 생성 (스트리밍 MSE) ─────────────────────────────────
async function generate() {
  let text = document.getElementById('text-input').value.trim();
  if (!text) return;

  const genBtn      = document.getElementById('generate-btn');
  const statusEl    = document.getElementById('gen-status');
  const videoEl     = document.getElementById('video-output');
  const placeholder = document.getElementById('video-placeholder');

  genBtn.disabled      = true;
  statusEl.textContent = '생성 중...';

  try {
    statusEl.textContent = 'AI 답변 생성 중...';
    text = await generateLLMReply(text);
  } catch (e) {
    statusEl.textContent = `OpenAI 오류: ${e.message}`;
    genBtn.disabled = !isModelReady;
    return;
  }

  const form = new FormData();
  form.append('text',  text);
  form.append('voice', document.getElementById('voice-select').value);

  const MIME   = 'video/mp4; codecs="avc1.42E01E, mp4a.40.2"';
  const useMSE = 'MediaSource' in window && MediaSource.isTypeSupported(MIME);

  if (useMSE) {
    await _generateStream(form, MIME, videoEl, placeholder, statusEl);
  } else {
    await readSSE('/api/generate', form, ({ status, error, video_path }) => {
      if (status)     statusEl.textContent = status;
      if (error)      statusEl.textContent = `오류: ${error}`;
      if (video_path) {
        videoEl.src               = `/api/video?path=${encodeURIComponent(video_path)}`;
        videoEl.style.display     = 'block';
        placeholder.style.display = 'none';
        videoEl.play();
      }
    });
  }

  genBtn.disabled = false;
}

async function generateLLMReply(text) {
  const response = await fetch('/api/llm/respond', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ text })
  });

  let data = {};
  try {
    data = await response.json();
  } catch (_) {}

  if (!response.ok) {
    throw new Error(data.detail || '답변 생성에 실패했습니다.');
  }

  return data.reply;
}

async function generateChat() {
  const input       = document.getElementById('text-input');
  const genBtn      = document.getElementById('generate-btn');
  const statusEl    = document.getElementById('gen-status');
  const videoEl     = document.getElementById('video-output');
  const placeholder = document.getElementById('video-placeholder');
  const userText    = input.value.trim();

  if (!userText || genBtn.disabled) return;

  appendChatMessage('user', userText);
  input.value = '';
  genBtn.disabled = true;

  try {
    statusEl.textContent = 'AI 답변 생성 중...';
    const reply = await generateLLMReply(userText);
    appendChatMessage('assistant', reply);

    statusEl.textContent = '아바타 영상 생성 중...';
    const form = new FormData();
    form.append('text', reply);
    form.append('voice', document.getElementById('voice-select').value);

    const MIME   = 'video/mp4; codecs="avc1.42E01E, mp4a.40.2"';
    const useMSE = 'MediaSource' in window && MediaSource.isTypeSupported(MIME);

    if (useMSE) {
      await _generateStream(form, MIME, videoEl, placeholder, statusEl);
    } else {
      await readSSE('/api/generate', form, ({ status, error, video_path }) => {
        if (status) statusEl.textContent = status;
        if (error) {
          statusEl.textContent = `오류: ${error}`;
          appendChatMessage('system', error);
        }
        if (video_path) {
          videoEl.src               = `/api/video?path=${encodeURIComponent(video_path)}`;
          videoEl.style.display     = 'block';
          placeholder.style.display = 'none';
          videoEl.play();
        }
      });
    }
  } catch (e) {
    statusEl.textContent = `오류: ${e.message}`;
    appendChatMessage('system', e.message);
  } finally {
    genBtn.disabled = !isModelReady;
    input.focus();
  }
}

function appendChatMessage(role, text) {
  const log = document.getElementById('chat-log');
  if (!log) return;

  const row = document.createElement('div');
  row.className = `message-row ${role}`;

  const time = document.createElement('div');
  time.className = 'message-time';
  time.textContent = '방금 전';

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  bubble.textContent = text;

  row.appendChild(time);
  row.appendChild(bubble);
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
}

async function _generateStream(form, mime, videoEl, placeholder, statusEl) {
  const mediaSource = new MediaSource();
  const objectURL   = URL.createObjectURL(mediaSource);

  videoEl.src               = objectURL;
  videoEl.style.display     = 'block';
  placeholder.style.display = 'none';

  await new Promise(resolve =>
    mediaSource.addEventListener('sourceopen', resolve, { once: true })
  );

  const sb          = mediaSource.addSourceBuffer(mime);
  const appendQueue = [];
  let   appending   = false;
  let   streamDone  = false;

  function flushQueue() {
    if (appending || appendQueue.length === 0 || mediaSource.readyState !== 'open') return;
    appending = true;
    sb.appendBuffer(appendQueue.shift());
  }

  sb.addEventListener('updateend', () => {
    appending = false;
    if (streamDone && appendQueue.length === 0) {
      try { mediaSource.endOfStream(); statusEl.textContent = '완료!'; } catch (_) {}
    } else {
      flushQueue();
    }
  });
  sb.addEventListener('error', e => console.error('[MSE]', e));

  // 버퍼 직접 모니터링
  const INITIAL_WAIT     = 0;
  const PAUSE_THRESHOLD  = 0.3;
  const RESUME_THRESHOLD = 1.5;
  let monitorId  = null;
  let started    = false;
  let playAllowed = false;

  function monitorBuffer() {
    if (!started) return;
    const buf = videoEl.buffered;
    if (buf.length > 0) {
      const ahead = buf.end(buf.length - 1) - videoEl.currentTime;
      if (!videoEl.paused && ahead < PAUSE_THRESHOLD) {
        videoEl.pause();
        statusEl.textContent = '버퍼링 중...';
      } else if (videoEl.paused && playAllowed && ahead >= RESUME_THRESHOLD) {
        videoEl.play().catch(() => {});
        statusEl.textContent = '재생 중...';
      }
    }
    monitorId = setTimeout(monitorBuffer, 200);
  }

  const response = await fetch('/api/generate_stream', {
    method:  'POST',
    body:    form,
    headers: { 'Authorization': `Bearer ${localStorage.getItem('access_token')}` }
  });
  const reader = response.body.getReader();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    appendQueue.push(value);
    flushQueue();

    if (!started) {
      started = true;
      statusEl.textContent = '버퍼링 중...';
      monitorBuffer();
      setTimeout(() => { playAllowed = true; }, INITIAL_WAIT);
    }
  }

  streamDone = true;
  if (!appending && appendQueue.length === 0 && mediaSource.readyState === 'open') {
    try { mediaSource.endOfStream(); statusEl.textContent = '완료!'; } catch (_) {}
  }
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

// ── SSE 유틸 ─────────────────────────────────────────────────
async function readSSE(url, formData, onMessage) {
  const res    = await fetch(url, {
    method: 'POST',
    body:   formData,
    headers: { 'Authorization': `Bearer ${localStorage.getItem('access_token')}` }
  });
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
