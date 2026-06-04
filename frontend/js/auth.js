const API = "";  // same origin (port 7862)

const loginView   = document.getElementById("loginView");
const signupView  = document.getElementById("signupView");
const loginMsg    = document.getElementById("loginMessage");
const signupMsg   = document.getElementById("signupMessage");

let isIdChecked = false;

// ── 화면 전환 ────────────────────────────────────────────────
document.getElementById("showSignupBtn").addEventListener("click", () => {
  loginView.classList.add("hidden");
  signupView.classList.remove("hidden");
  document.getElementById("signupForm").reset();
  signupMsg.textContent = "";
  isIdChecked = false;
});

document.getElementById("showLoginBtn").addEventListener("click", () => {
  signupView.classList.add("hidden");
  loginView.classList.remove("hidden");
  document.getElementById("loginForm").reset();
  loginMsg.textContent = "";
});

document.getElementById("signupUsername").addEventListener("input", () => {
  isIdChecked = false;
});

// ── 중복 확인 ────────────────────────────────────────────────
document.getElementById("checkIdBtn").addEventListener("click", async () => {
  const username = document.getElementById("signupUsername").value.trim();
  if (!username) {
    setMsg(signupMsg, "아이디를 먼저 입력해주세요.", false);
    return;
  }
  try {
    const data = await fetch(`${API}/api/check-username?username=${username}`).then(r => r.json());
    if (data.is_available) {
      setMsg(signupMsg, "사용 가능한 아이디입니다.", true);
      isIdChecked = true;
    } else {
      setMsg(signupMsg, "이미 사용 중인 아이디입니다.", false);
      isIdChecked = false;
    }
  } catch { setMsg(signupMsg, "서버 통신 에러", false); }
});

// ── 회원가입 ─────────────────────────────────────────────────
document.getElementById("signupForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!isIdChecked) { setMsg(signupMsg, "아이디 중복확인을 먼저 해주세요.", false); return; }

  const username = document.getElementById("signupUsername").value;
  const password = document.getElementById("signupPassword").value;
  const confirm  = document.getElementById("signupPasswordConfirm").value;

  if (password !== confirm) { setMsg(signupMsg, "비밀번호가 일치하지 않습니다.", false); return; }

  try {
    const res = await fetch(`${API}/api/signup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    const data = await res.json();
    if (res.ok) {
      setMsg(signupMsg, "회원가입 완료! 로그인 해주세요.", true);
      setTimeout(() => {
        document.getElementById("showLoginBtn").click();
        document.getElementById("loginUsername").value = username;
      }, 800);
    } else {
      setMsg(signupMsg, data.detail || "회원가입 실패", false);
    }
  } catch { setMsg(signupMsg, "서버 통신 에러", false); }
});

// ── 로그인 ───────────────────────────────────────────────────
document.getElementById("loginForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const username = document.getElementById("loginUsername").value;
  const password = document.getElementById("loginPassword").value;

  try {
    const res = await fetch(`${API}/api/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    const data = await res.json();
    if (res.ok) {
      localStorage.setItem("access_token", data.access_token);
      localStorage.setItem("username", username);
      window.location.href = "/";
    } else {
      setMsg(loginMsg, "아이디 또는 비밀번호가 틀렸습니다.", false);
    }
  } catch { setMsg(loginMsg, "서버 통신 에러", false); }
});

// ── 유틸 ─────────────────────────────────────────────────────
function setMsg(el, text, ok) {
  el.textContent = text;
  el.style.color = ok ? "#4ade80" : "#ff6b6b";
}
