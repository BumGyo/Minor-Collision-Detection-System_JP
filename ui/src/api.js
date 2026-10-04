// FastAPI 백엔드 연동 클라이언트.
// vite 프록시(/api → http://localhost:8000)를 통해 호출한다.

const TOKEN_KEY = "psd_token";
const USER_KEY = "psd_user";

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}
export function getStoredUser() {
  const raw = localStorage.getItem(USER_KEY);
  return raw ? JSON.parse(raw) : null;
}
export function saveAuth(token, user) {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}
export function clearAuth() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

async function request(path, { method = "GET", body, isForm = false, signal } = {}) {
  const headers = {};
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  let payload = body;
  if (body && !isForm) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }
  const res = await fetch(`/api${path}`, { method, headers, body: payload, signal });
  if (!res.ok) {
    const statusMessages = {
      400: "入力内容を確認してください。",
      401: "ユーザーIDまたはパスワードが正しくありません。",
      403: "この操作を実行する権限がありません。",
      404: "対象のデータが見つかりません。",
      500: "サーバーでエラーが発生しました。",
    };
    let detail = statusMessages[res.status] || `リクエストに失敗しました（${res.status}）`;
    try {
      const data = await res.json();
         if (res.status !== 401 && !/[가-힣]/.test(data.detail || "")) {
        detail = data.detail || detail;
      }
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  if (res.status === 204) return null;
  return res.json();
}

// ---------- 데이터 정규화 (API → UI 형태) ----------
function normalizeEvent(e) {
  return {
    id: e.id,
    timestamp: Math.round(e.timestamp_sec || 0),
    endTimestamp:
      e.end_timestamp_sec != null ? Math.round(e.end_timestamp_sec) : null,
    prob: e.crash_prob,
    hasClip: !!e.has_clip,
    title: "事故の疑いがある区間",
    status:
      e.crash_prob != null && e.crash_prob >= 0.5 ? "確認が必要" : "分析完了",
  };
}

export function normalizeVideo(v) {
  const created = v.created_at || "";
  return {
    id: v.id,
    date: v.recording_date || (created ? created.slice(0, 10) : ""),
    startTime: v.recording_start_time || "20:30",
    camera: v.camera_location || "駐車場",
    duration: Math.round(v.duration_sec || 0),
    width: v.width,
    height: v.height,
    fps: v.fps,
    events: (v.events || []).map(normalizeEvent),
  };
}

// ---------- 인증 ----------
export const api = {
  signup: (data) => request("/auth/signup", { method: "POST", body: data }),
  login: (data) => request("/auth/login", { method: "POST", body: data }),
  getMe: () => request("/auth/me"),
  updateMe: (data) => request("/auth/me", { method: "PUT", body: data }),
  changePassword: (data) => request("/auth/change-password", { method: "POST", body: data }),

  // ---------- 영상 ----------
  listVideos: async (days) => {
    const q = days ? `?days=${days}` : "";
    const list = await request(`/videos${q}`);
    return list.map(normalizeVideo);
  },
  getVideo: async (id) => normalizeVideo(await request(`/videos/${id}`)),
  uploadVideo: (file, recordingDate) => {
    const form = new FormData();
    form.append("file", file);
    if (recordingDate) form.append("recording_date", recordingDate);
    return request("/videos", { method: "POST", body: form, isForm: true });
  },
  streamUrl: (id) => `/api/videos/${id}/stream`,
  thumbnailUrl: (id) => `/api/videos/${id}/thumbnail`,
  deleteVideo: (id) => request(`/videos/${id}`, { method: "DELETE" }),
  deleteEvent: (videoId, eventId) =>
    request(`/videos/${videoId}/events/${eventId}`, { method: "DELETE" }),
  clearEvents: (videoId) =>
    request(`/videos/${videoId}/events`, { method: "DELETE" }),

  // ---------- 분석 ----------
  detectVehicles: (videoId, timestampSec = 0, signal) =>
    request(`/videos/${videoId}/detect-vehicles?timestamp_sec=${timestampSec}`, {
      method: "POST",
      signal,
    }),
  analyze: (videoId, bbox, signal) =>
    request(`/videos/${videoId}/analyze`, { method: "POST", body: bbox, signal }),
  taskStatus: (taskId) => request(`/tasks/${taskId}`),
  cancelTask: (taskId) => request(`/tasks/${taskId}/cancel`, { method: "POST" }),
  clipUrl: (eventId) => `/api/events/${eventId}/clip`,
};



