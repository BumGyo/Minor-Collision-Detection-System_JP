// FastAPIバックエンド連携クライアント。
// Viteプロキシ(/api → http://localhost:8000)経由で呼び出す。

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

export async function retryRequest(operation, { attempts = 3, delayMs = 500 } = {}) {
  let lastError;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;
      if (attempt < attempts) {
        await new Promise((resolve) => setTimeout(resolve, delayMs * attempt));
      }
    }
  }
  throw lastError;
}

export async function withRequestTimeout(operation, timeoutMs) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await operation(controller.signal);
  } finally {
    clearTimeout(timeout);
  }
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
  // ログインの有効期限が切れたかトークンが無効な場合、保存されたログイン情報を消去してログイン画面へ。
  // (ログイン・会員登録自体の401 — ID/パスワード誤り — は画面にそのまま表示する)
  if (res.status === 401 && !path.startsWith("/auth/login") && !path.startsWith("/auth/signup")) {
    clearAuth();
    if (!window.location.pathname.startsWith("/login")) {
      window.location.replace("/login?expired=1");
    }
    throw new Error("ログインの有効期限が切れました。再度ログインしてください。");
  }
  if (!res.ok) {
    let detail = `リクエスト失敗 (${res.status})`;
    try {
      const data = await res.json();
      detail = data.detail || detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  if (res.status === 204) return null;
  return res.json();
}

// ---------- データ正規化 (API → UI形式) ----------
function normalizeEvent(e) {
  return {
    id: e.id,
    timestamp: Math.round(e.timestamp_sec || 0),
    endTimestamp:
      e.end_timestamp_sec != null ? Math.round(e.end_timestamp_sec) : null,
    // 四捨五入前の値 — 再生バーの事故区間位置・移動など1秒より高い精度が必要な箇所で使用
    startSec: e.timestamp_sec || 0,
    endSec: e.end_timestamp_sec ?? null,
    analyzedAt: e.analyzed_at || null, // 分析が完了してクリップが生成された時刻(ISO, UTC)
    analysisNo: e.analysis_no ?? null, // この動画の何回目の分析から抽出されたクリップか (1回の分析 = 同一番号)
    clipStartSec: e.clip_start_sec ?? null, // クリップが元動画において開始する時刻
    prob: e.crash_prob,
    hasClip: !!e.has_clip,
    hasRawClip: !!e.has_raw_clip, // オーバーレイなしの原本事故クリップをダウンロード可能か
    title: "事故疑い区間",
    status:
      e.crash_prob != null && e.crash_prob >= 0.5 ? "要確認" : "分析完了",
  };
}

export function normalizeVideo(v) {
  const created = v.created_at || "";
  return {
    id: v.id,
    title: v.title || "", // アップロード時に入力されたタイトル (指定がない場合は画面側で '録画日 録画データ #N')
    fileName: v.video_name || "", // アップロードされた原本ファイル名
    uploadedAt: v.created_at || null, // アップロード時刻(ISO, UTC)
    // 保存期間が終了して動画・クリップが削除されるまでの残り日数(サーバー時計基準)。保存期間無効時はnull
    expiresInDays: v.expires_in_days ?? null,
    date: v.recording_date || (created ? created.slice(0, 10) : ""),
    duration: Math.round(v.duration_sec || 0),
    width: v.width,
    height: v.height,
    fps: v.fps,
    events: (v.events || []).map(normalizeEvent),
    // 分析を一度でも完了したか — 再分析時の確認メッセージを切り替える
    analyzedBefore: !!v.analysis_done,
    // 直近の分析タスク — 失敗・中断時は画面に「再実行」を案内
    lastTask: v.last_task
      ? { id: v.last_task.task_id, status: v.last_task.status, errorMessage: v.last_task.error_message }
      : null,
    // 分析履歴(リクエスト順) — 動画詳細のみで取得。時刻はISO(UTC)
    analysisHistory: (v.analysis_history || []).map((h) => ({
      no: h.analysis_no,
      taskId: h.task_id,
      status: h.status,
      errorMessage: h.error_message,
      requestedAt: h.requested_at,
      finishedAt: h.finished_at,
      eventCount: h.event_count, // 検出されたクリップ数 (履歴機能実装前の分析はnull)
    })),
  };
}

// ---------- 認証 ----------
export const api = {
  signup: (data) => request("/auth/signup", { method: "POST", body: data }),
  login: (data) => request("/auth/login", { method: "POST", body: data }),
  // 画面リフレッシュ時 — 最新のユーザー情報 + 動画・サムネイル・クリップ用Cookie(ログイン時にも受領)
  startSession: () => request("/auth/session", { method: "POST" }),
  logout: () => request("/auth/logout", { method: "POST" }),
  updateMe: (data) => request("/auth/me", { method: "PUT", body: data }),
  changePassword: (data) => request("/auth/change-password", { method: "POST", body: data }),

  // ---------- 動画 ----------
  listVideos: async (days) => {
    const q = days ? `?days=${days}` : "";
    const list = await retryRequest(() =>
      withRequestTimeout((signal) => request(`/videos${q}`, { signal }), 7000));
    return list.map(normalizeVideo);
  },
  getVideo: async (id) => normalizeVideo(await request(`/videos/${id}`)),
  uploadVideo: (file, recordingDate, title) => {
    const form = new FormData();
    form.append("file", file);
    if (recordingDate) form.append("recording_date", recordingDate);
    if (title && title.trim()) form.append("title", title.trim());
    return request("/videos", { method: "POST", body: form, isForm: true });
  },
  streamUrl: (id) => `/api/videos/${id}/stream`,
  thumbnailUrl: (id) => `/api/videos/${id}/thumbnail`,
  deleteVideo: (id) => request(`/videos/${id}`, { method: "DELETE" }),
  clearAnalysisHistory: (id) => request(`/videos/${id}/analysis-history`, { method: "DELETE" }),
  deleteEvent: (videoId, eventId) =>
    request(`/videos/${videoId}/events/${eventId}`, { method: "DELETE" }),
  clearEvents: (videoId) =>
    request(`/videos/${videoId}/events`, { method: "DELETE" }),

  // ---------- 分析 ----------
  detectVehicles: (videoId, timestampSec = 0, signal) =>
    request(`/videos/${videoId}/detect-vehicles?timestamp_sec=${timestampSec}`, {
      method: "POST",
      signal,
    }),
  getVehicleDetectionTask: (videoId, taskId, signal) =>
    request(`/videos/${videoId}/vehicle-detection-tasks/${taskId}`, { signal }),
  analyze: (videoId, bbox, signal) =>
    request(`/videos/${videoId}/analyze`, { method: "POST", body: bbox, signal }),
  getTaskStatus: (taskId) => request(`/tasks/${taskId}`),
  cancelTask: (taskId) => request(`/tasks/${taskId}/cancel`, { method: "POST" }),
  // ログイン中のユーザーの待機・進行中タスク(画面再表示時の進捗復元用)
  listActiveTasks: () => request("/tasks/active"),
  // 失敗・キャンセルされた分析を同じ動画・同じ車両ボックスで再実行
  retryTask: (taskId) => request(`/tasks/${taskId}/retry`, { method: "POST" }),
  clipUrl: (eventId) => `/api/events/${eventId}/clip`,
  // ボックス・ステータスバーなしで事故区間のみ切り出した原本クリップ(ダウンロード用)
  rawClipUrl: (eventId) => `/api/events/${eventId}/clip/raw`,
};
