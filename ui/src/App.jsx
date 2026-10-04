import { useMemo, useRef, useState, useEffect } from "react";
import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
  useNavigate,
  useParams,
} from "react-router-dom";
import { api, saveAuth, clearAuth, getToken, getStoredUser } from "./api";
import "./App.css";

function AppLoadingScreen() {
  return (
    <div className="app-splash-screen">
      <div className="app-splash-bg">
        <div className="cosmic-stars" />
        <div className="cosmic-glow-limb" />
      </div>
      <div className="app-splash-content">
        <img className="app-splash-logo" src="/siot-logo-loading.svg" alt="SIOT" />
        <p className="app-splash-subtitle">駐車事故イベント検知システム</p>
        <div className="app-splash-spinner-ring" />
      </div>
    </div>
  );
}

function LoginPage({ onLogin }) {
  const [id, setId] = useState("");
  const [pw, setPw] = useState("");
  const [error, setError] = useState("");
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const [isSignupOpen, setIsSignupOpen] = useState(false);
  const navigate = useNavigate();

  const handleLogin = async () => {
    setError("");
    try {
      const res = await api.login({ username: id, password: pw });
      saveAuth(res.token, res.user);
      onLogin(res.user);
      navigate("/videos");
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <div className="public-login-page">
      <header className="public-navbar">
        <button className="public-brand" onClick={() => navigate("/login")}>
          <img className="public-brand-logo" src="/siot-logo.svg" alt="SIOT" />
        </button>
        <div className="public-nav-meta">
          <span className="public-nav-label">Parking incident intelligence</span>
          <button className="public-login-trigger" onClick={() => setIsLoginOpen(true)}>
            LOGIN <span aria-hidden="true">↗</span>
          </button>
        </div>
      </header>

      <main className="public-main">
        <section className="public-hero">
          <div className="public-hero-copy">
            <p className="public-kicker">CCTV EVENT REVIEW / 01</p>
            <h1>管理者がすべての映像を<br /><em>確認しなくても、AIが先に見つけます。</em></h1>
            <p className="public-hero-description">
              当て逃げの疑いがある場面を映像から見つけ、日付とイベント単位で整理します。
              分散したCCTV記録を、確認できる事故データに変えるモニタリングシステムです。
            </p>
            <div className="public-hero-actions">
              <button className="public-primary-btn" onClick={() => setIsLoginOpen(true)}>
                ダッシュボードを開く <span aria-hidden="true">→</span>
              </button>
              <a className="public-text-link" href="#why-siot">必要な理由 <span aria-hidden="true">↓</span></a>
            </div>
          </div>

          <div className="public-hero-panel" aria-label="SIOTの分析フロー">
            <div className="public-panel-topline">
              <span>LIVE REVIEW SYSTEM</span>
              <span className="public-live-dot">● ONLINE</span>
            </div>
            <div className="public-scan-frame">
              <div className="public-scan-grid" />
              <div className="public-scan-road">
                <span className="public-car car-one" />
                <span className="public-car car-two" />
                <span className="public-car car-three" />
              </div>
              <div className="public-detection-box box-one"><span>VEHICLE 01</span></div>
              <div className="public-detection-box box-two"><span>EVENT CHECK</span></div>
              <div className="public-scan-caption">FRAME 004281 / CAMERA C1</div>
            </div>
            <div className="public-panel-footer">
              <span>疑わしい区間を検知</span>
              <strong>→ 日付別イベントに整理</strong>
            </div>
          </div>
        </section>

        <section className="public-proof-strip" aria-label="SIOTの主な機能">
          <div><strong>01</strong><span>映像をアップロード</span></div>
          <div><strong>02</strong><span>疑わしい場面を分析</span></div>
          <div><strong>03</strong><span>案件ごとに確認</span></div>
          <p>AIが映像を先に確認し、管理者は必要な場面に集中できます。</p>
        </section>

        <section className="public-insight-section" id="why-siot">
          <div className="public-section-heading">
            <p className="public-kicker">RECENT HIT-AND-RUN TREND</p>
            <h2>繰り返される当て逃げ、<br /><em>AIによる選別が必要な理由</em></h2>
            <p>
              公開された警察統計と最近の報道から確認できる傾向をまとめました。
              SIOTは膨大なCCTV映像から確認が必要な瞬間を先に絞り、管理者の時間を減らします。
            </p>
          </div>
          <div className="public-trend-table-wrap">
            <div className="public-trend-table-heading">
              <span>INDICATOR</span><span>OBSERVED DATA</span><span>WHAT IT MEANS</span>
            </div>
            <div className="public-trend-row highlight">
              <strong>全国 / 2024</strong>
              <b>約7万件</b>
              <span>警察庁の交通取締統計を引用した、2024年の年間届出件数</span>
            </div>
            <div className="public-trend-row">
              <strong>光州 / 2021</strong>
              <b>約9,600件</b>
              <span>地域警察の集計で、事故後の措置を取らず立ち去った事案が1日約25件</span>
            </div>
            <div className="public-trend-row">
              <strong>映像確保の期限</strong>
              <b>2〜4週間</b>
              <span>駐車場CCTVは上書きされる前に必要な区間を確保する必要があります</span>
            </div>
            <small className="public-trend-source">出典：2026年6月18日ロトクニュースによる警察庁交通取締統計2024の引用、2022年10月24日光州放送による光州警察集計。CCTV保存期間はロトクニュースの案内を参照。</small>
          </div>
        </section>
      </main>

      {isLoginOpen && (
        <div className="login-modal-overlay" onClick={() => setIsLoginOpen(false)}>
          <div className="login-modal" role="dialog" aria-modal="true" aria-labelledby="login-modal-title" onClick={(e) => e.stopPropagation()}>
            <button className="login-modal-close" onClick={() => setIsLoginOpen(false)} aria-label="ログイン画面を閉じる">×</button>
            <p className="public-kicker">SECURE ACCESS</p>
            <h2 id="login-modal-title">ダッシュボードにログイン</h2>
            <p className="login-modal-description">分析済みの映像と事故の疑いがあるイベントを確認します。</p>

            <div className="input-group">
              <input
                type="text"
                placeholder="ユーザーID"
                value={id}
                onChange={(e) => setId(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleLogin()}
                autoFocus
              />
            </div>

            <div className="input-group">
              <input
                type="password"
                placeholder="パスワード"
                value={pw}
                onChange={(e) => setPw(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleLogin()}
              />
            </div>

            {error && <p className="login-error-text">{error}</p>}

            <div className="login-options">
              <label className="remember-me">
                <input type="checkbox" /> ログイン状態を保持
              </label>
              <a href="#none" className="support-link">Support</a>
            </div>

            <button className="login-submit-btn" onClick={handleLogin}>ログイン</button>
            <button
              className="login-signup-btn"
              onClick={() => {
                setIsLoginOpen(false);
                setIsSignupOpen(true);
              }}
            >
              アカウント登録
            </button>
          </div>
        </div>
      )}

      {isSignupOpen && <SignupModal onClose={() => setIsSignupOpen(false)} onBackToLogin={() => {
        setIsSignupOpen(false);
        setIsLoginOpen(true);
      }} />}
    </div>
  );
}

function SignupModal({ onClose, onBackToLogin }) {
  const [form, setForm] = useState({
    username: "",
    name: "",
    email: "",
    password: "",
    passwordConfirm: "",
  });
  const [error, setError] = useState("");
  const [done, setDone] = useState(false);

  const update = (key) => (e) =>
    setForm((prev) => ({ ...prev, [key]: e.target.value }));

  const handleSignup = async () => {
    setError("");
    if (!form.username || !form.name || !form.password) {
      setError("ユーザーID、氏名、パスワードは必須です。");
      return;
    }
    if (form.password !== form.passwordConfirm) {
      setError("パスワードが一致しません。");
      return;
    }
    try {
      await api.signup({
        username: form.username,
        name: form.name,
        email: form.email || null,
        password: form.password,
      });
      setDone(true);
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <div className="login-modal-overlay" onClick={onClose}>
      <div className="login-modal signup-modal" role="dialog" aria-modal="true" aria-labelledby="signup-modal-title" onClick={(e) => e.stopPropagation()}>
        <button className="login-modal-close" onClick={onClose} aria-label="登録画面を閉じる">×</button>
        <p className="public-kicker">CREATE ACCOUNT</p>
        <h2 id="signup-modal-title">アカウント登録</h2>
        <p className="login-modal-description">SIOT映像分析ダッシュボードのアカウントを作成します。</p>

        {done ? (
          <div className="signup-complete-state">
            <strong>登録が完了しました。</strong>
            <span>ログインしてダッシュボードを確認できます。</span>
            <button className="login-submit-btn" onClick={onBackToLogin}>ログインする</button>
          </div>
        ) : (
          <>
            <div className="signup-modal-grid">
              <div className="signup-field">
                <label className="signup-field-label" htmlFor="signup-name">氏名 <span>*</span></label>
                <div className="input-group">
                  <input id="signup-name" type="text" placeholder="氏名を入力してください" value={form.name} onChange={update("name")} autoFocus />
                </div>
              </div>
              <div className="signup-field">
                <label className="signup-field-label" htmlFor="signup-email">メールアドレス</label>
                <div className="input-group">
                  <input id="signup-email" type="email" placeholder="メールアドレスを入力してください" value={form.email} onChange={update("email")} autoCapitalize="off" autoCorrect="off" />
                </div>
              </div>
              <div className="signup-field">
                <label className="signup-field-label" htmlFor="signup-username">ユーザーID <span>*</span></label>
                <div className="input-group">
                  <input id="signup-username" type="text" placeholder="ログインに使用するID" value={form.username} onChange={update("username")} autoCapitalize="off" autoCorrect="off" />
                </div>
              </div>
              <div className="signup-field">
                <label className="signup-field-label" htmlFor="signup-password">パスワード <span>*</span></label>
                <div className="input-group">
                  <input id="signup-password" type="password" placeholder="パスワードを入力してください" value={form.password} onChange={update("password")} />
                </div>
              </div>
              <div className="signup-field">
                <label className="signup-field-label" htmlFor="signup-password-confirm">パスワード確認 <span>*</span></label>
                <div className="input-group">
                  <input id="signup-password-confirm" type="password" placeholder="パスワードをもう一度入力してください" value={form.passwordConfirm} onChange={update("passwordConfirm")} />
                </div>
              </div>
            </div>
            {error && <p className="login-error-text">{error}</p>}
            <button className="login-submit-btn" onClick={handleSignup}>登録する</button>
            <button className="login-signup-btn" onClick={onBackToLogin}>ログインに戻る</button>
          </>
        )}
      </div>
    </div>
  );
}

function SignupPage() {
  const [form, setForm] = useState({
    username: "",
    name: "",
    email: "",
    password: "",
    passwordConfirm: "",
  });
  const [error, setError] = useState("");
  const [done, setDone] = useState(false);
  const navigate = useNavigate();

  const update = (key) => (e) =>
    setForm((prev) => ({ ...prev, [key]: e.target.value }));

  const handleSignup = async () => {
    setError("");
    if (!form.username || !form.name || !form.password) {
      setError("ユーザーID、氏名、パスワードは必須です。");
      return;
    }
    if (form.password !== form.passwordConfirm) {
      setError("パスワードが一致しません。");
      return;
    }
    try {
      await api.signup({
        username: form.username,
        name: form.name,
        email: form.email || null,
        password: form.password,
      });
      setDone(true);
      setTimeout(() => navigate("/login"), 1200);
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <div className="login-layout">
      <div className="login-left-panel">
        <div className="brand-content">
          <div className="brand-badge">Parking Scratch Detection</div>
          <h1>アカウント登録</h1>
          <p>管理者アカウントを作成してシステムにアクセスしてください。</p>
        </div>
      </div>

      <div className="login-right-panel">
        <div className="login-form-wrapper">
          <h2>アカウント登録</h2>
          <p className="login-subtitle">新しいアカウントを作成します。</p>

          <div className="input-group">
            <input type="text" placeholder="氏名 *" value={form.name} onChange={update("name")}
              lang="ko" style={{ imeMode: "active" }} />
          </div>
          <div className="input-group">
            <input type="text" placeholder="ユーザーID *" value={form.username} onChange={update("username")}
              lang="en" style={{ imeMode: "inactive" }} autoCapitalize="off" autoCorrect="off" />
          </div>
          <div className="input-group">
            <input type="email" placeholder="メールアドレス" value={form.email} onChange={update("email")}
              lang="en" style={{ imeMode: "inactive" }} autoCapitalize="off" autoCorrect="off" />
          </div>
          <div className="input-group">
            <input type="password" placeholder="パスワード *" value={form.password} onChange={update("password")}
              lang="en" style={{ imeMode: "inactive" }} />
          </div>
          <div className="input-group">
            <input type="password" placeholder="パスワード確認 *" value={form.passwordConfirm} onChange={update("passwordConfirm")}
              lang="en" style={{ imeMode: "inactive" }} />
          </div>

          {error && <p className="login-error-text">{error}</p>}
          {done && <p className="login-success-text">登録完了。ログイン画面へ移動します。</p>}

          <button className="login-submit-btn" onClick={handleSignup}>登録する</button>
          <button className="login-signup-btn" onClick={() => navigate("/login")}>
            ログインに戻る
          </button>
        </div>
      </div>
    </div>
  );
}

// 유틸 함수들
function formatDate(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function addDays(date, days) {
  const copied = new Date(date);
  copied.setDate(copied.getDate() + days);
  return copied;
}

function formatTime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function formatActualTime(dateText, startTime, seconds) {
  const [hours, minutes] = startTime.split(":").map(Number);
  const date = new Date(`${dateText}T00:00:00`);
  date.setHours(hours, minutes, seconds, 0);

  return date.toLocaleTimeString("ja-JP", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function getVideoDisplayTitle(video, allVideos) {
  if (!video) return "";
  const sameDateVideos = (allVideos || []).filter((v) => v.date === video.date);
  if (sameDateVideos.length > 1) {
    const sorted = [...sameDateVideos].sort((a, b) => a.id - b.id);
    const index = sorted.findIndex((v) => v.id === video.id);
    const num = index >= 0 ? index + 1 : 1;
    return `${video.date} ${video.camera} 録画 #${num}`;
  }
  return `${video.date} ${video.camera} 録画`;
}

function formatMonthLabel(date) {
  return `${date.getFullYear()}年${date.getMonth() + 1}月`;
}

function getCalendarDays(monthDate) {
  const year = monthDate.getFullYear();
  const month = monthDate.getMonth();
  const firstDay = new Date(year, month, 1);
  const lastDay = new Date(year, month + 1, 0);
  const days = [];

  for (let index = 0; index < firstDay.getDay(); index += 1) {
    days.push(null);
  }

  for (let day = 1; day <= lastDay.getDate(); day += 1) {
    days.push(new Date(year, month, day));
  }

  return days;
}

function getVideosByDateApi(videos) {
  return videos.reduce((groups, video) => {
    groups[video.date] = [...(groups[video.date] ?? []), video];
    return groups;
  }, {});
}

// 2. 이벤트 통계 분석 뷰 (Custom SVG Charts)
function AnalyticsView({ filteredVideos, filterDays, setFilterDays }) {
  const stats = useMemo(() => {
    let totalEvents = 0;
    let pendingCount = 0;
    let resolvedCount = 0;

    const dateCounts = {};
    const cameraCounts = {};
    const eventTypeCounts = {};
    const hourCounts = Array(24).fill(0);

    // filteredVideos는 이미 API에서 정규화된 영상 목록
    const matchingVideos = filteredVideos;

    matchingVideos.forEach(video => {
      const [startHour] = video.startTime.split(":").map(Number);

      video.events.forEach(event => {
        totalEvents += 1;

        // Status counts
        if (event.status === "確認が必要") pendingCount++;
        else if (event.status === "分析完了") resolvedCount++;

        // Date grouping
        dateCounts[video.date] = (dateCounts[video.date] || 0) + 1;

        // Camera grouping
        cameraCounts[video.camera] = (cameraCounts[video.camera] || 0) + 1;

        // Type grouping
        let type = "その他";
        if (event.title.includes("スクラッチ") || event.title.includes("擦り傷")) type = "擦り傷の疑い";
        else if (event.title.includes("ドアパンチ")) type = "ドアパンチの疑い";
        else if (event.title.includes("衝突") || event.title.includes("接触")) type = "車両衝突の疑い";
        else if (event.title.includes("接近") || event.title.includes("検知")) type = "人物接近を検知";

        eventTypeCounts[type] = (eventTypeCounts[type] || 0) + 1;

        // Hour computation
        const eventSeconds = event.timestamp;
        const eventHour = (startHour + Math.floor(eventSeconds / 3600)) % 24;
        hourCounts[eventHour]++;
      });
    });

    // Sort dates chronologically
    const sortedDates = Object.keys(dateCounts).sort().map(date => ({
      date: date.substring(5), // MM-DD
      count: dateCounts[date]
    }));

    // Convert cameras to array
    const cameraList = Object.keys(cameraCounts).map(cam => ({
      name: cam,
      count: cameraCounts[cam]
    })).sort((a, b) => b.count - a.count);

    // Convert event types to array
    const eventTypeList = Object.keys(eventTypeCounts).map(type => ({
      name: type,
      count: eventTypeCounts[type]
    }));

    // Hourly grouped brackets
    const hourlyGroups = [
      { name: "深夜 (00-06)", count: 0 },
      { name: "午前 (06-12)", count: 0 },
      { name: "午後 (12-18)", count: 0 },
      { name: "夜間 (18-24)", count: 0 }
    ];
    for (let h = 0; h < 24; h++) {
      const count = hourCounts[h];
      if (h < 6) hourlyGroups[0].count += count;
      else if (h < 12) hourlyGroups[1].count += count;
      else if (h < 18) hourlyGroups[2].count += count;
      else hourlyGroups[3].count += count;
    }

    return {
      totalVideos: matchingVideos.length,
      totalEvents,
      pendingCount,
      resolvedCount,
      sortedDates,
      cameraList,
      eventTypeList,
      hourlyGroups
    };
  }, [filteredVideos]);

  const dailyTrendChart = useMemo(() => {
    const width = 500;
    const height = 200;
    const paddingX = 40;
    const paddingY = 30;
    const plotW = width - paddingX * 2;
    const plotH = height - paddingY * 2;
    const data = stats.sortedDates;
    if (data.length === 0) return null;

    const maxVal = Math.max(...data.map(d => d.count), 4) + 1;

    const points = data.map((d, i) => {
      const x = paddingX + (i * plotW) / (data.length - 1);
      const y = height - paddingY - (d.count * plotH) / maxVal;
      return { x, y, label: d.date, value: d.count };
    });

    const linePath = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ");
    const areaPath = `${linePath} L ${points[points.length - 1].x} ${height - paddingY} L ${points[0].x} ${height - paddingY} Z`;

    return (
      <svg width="100%" height="200" viewBox={`0 0 ${width} ${height}`} className="stats-svg">
        <defs>
          <linearGradient id="area-gradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--chart-primary)" stopOpacity="0.4" />
            <stop offset="100%" stopColor="var(--chart-primary)" stopOpacity="0.0" />
          </linearGradient>
        </defs>

        {/* Y Axis Grid lines */}
        {[0, 1, 2, 3, 4, 5].map(v => {
          const y = height - paddingY - (v * plotH) / maxVal;
          return (
            <g key={v}>
              <line x1={paddingX} y1={y} x2={width - paddingX} y2={y} stroke="var(--chart-grid)" strokeDasharray="3 3" />
              <text x={paddingX - 10} y={y + 4} textAnchor="end" className="chart-axis-text" fill="var(--chart-text)">{v}</text>
            </g>
          );
        })}

        {/* Area */}
        <path d={areaPath} fill="url(#area-gradient)" />

        {/* Line */}
        <path d={linePath} fill="none" stroke="var(--chart-primary)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />

        {/* Dots */}
        {points.map((p, i) => (
          <g key={i}>
            <circle cx={p.x} cy={p.y} r="4" fill="var(--chart-primary)" stroke="var(--chart-grid)" strokeWidth="1" />
            <text x={p.x} y={p.y - 10} textAnchor="middle" className="chart-value-text" fill="var(--chart-text-primary)">
              {p.value}
            </text>
            <text x={p.x} y={height - paddingY + 16} textAnchor="middle" className="chart-axis-text" fill="var(--chart-text)">
              {p.label}
            </text>
          </g>
        ))}
      </svg>
    );
  }, [stats.sortedDates]);

  const donutChart = useMemo(() => {
    const total = stats.totalEvents;
    if (total === 0) return null;
    const r = 50;
    const circ = 2 * Math.PI * r;
    let currentAngle = -90;

    const isDark = typeof document !== "undefined" && document.body.classList.contains("dark-mode");
    const colors = isDark
      ? ["#00e699", "#06b6d4", "#f59e0b", "#ef4444", "#a855f7"]
      : ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"];

    const segments = [];
    for (let idx = 0; idx < stats.eventTypeList.length; idx++) {
      const type = stats.eventTypeList[idx];
      const percentage = (type.count / total) * 100;
      const angle = (type.count / total) * 360;
      const strokeDashoffset = circ - (type.count / total) * circ;
      const rotation = currentAngle;
      currentAngle += angle;
      const color = colors[idx % colors.length];

      segments.push({
        ...type,
        percentage: percentage.toFixed(1),
        strokeDashoffset,
        rotation,
        color
      });
    }

    return (
      <div className="donut-chart-container">
        <svg width="180" height="180" viewBox="0 0 200 200">
          <circle cx="100" cy="100" r={r} fill="transparent" stroke="var(--chart-grid)" strokeWidth="16" />
          {segments.map((seg, idx) => (
            <circle
              key={idx}
              cx="100"
              cy="100"
              r={r}
              fill="transparent"
              stroke={seg.color}
              strokeWidth="18"
              strokeDasharray={circ}
              strokeDashoffset={seg.strokeDashoffset}
              transform={`rotate(${seg.rotation} 100 100)`}
              className="donut-segment"
            />
          ))}
          <g className="donut-center-text">
            <text x="100" y="95" textAnchor="middle" className="donut-total" fill="var(--chart-text-primary)" style={{ fontSize: "28px", fontWeight: "800" }}>
              {total}
            </text>
            <text x="100" y="115" textAnchor="middle" className="donut-label" fill="var(--chart-text)" style={{ fontSize: "12px", fontWeight: "600" }}>
              検知件数
            </text>
          </g>
        </svg>

        <div className="donut-legend">
          {segments.map((seg, idx) => (
            <div key={idx} className="legend-item">
              <span className="legend-badge" style={{ backgroundColor: seg.color }} />
              <span className="legend-name">{seg.name}</span>
              <span className="legend-count">{seg.count}件（{seg.percentage}%）</span>
            </div>
          ))}
        </div>
      </div>
    );
  }, [stats.eventTypeList, stats.totalEvents]);

  const hourlyChartSvg = useMemo(() => {
    const width = 450;
    const height = 200;
    const paddingX = 40;
    const paddingY = 30;
    const plotW = width - paddingX * 2;
    const plotH = height - paddingY * 2;
    const data = stats.hourlyGroups;
    const maxVal = Math.max(...data.map(d => d.count), 4) + 1;

    const barW = 32;
    const gap = (plotW - barW * data.length) / (data.length - 1);

    return (
      <svg width="100%" height="200" viewBox={`0 0 ${width} ${height}`} className="stats-svg">
        {[0, 1, 2, 3, 4, 5].map(v => {
          const y = height - paddingY - (v * plotH) / maxVal;
          return (
            <g key={v}>
              <line x1={paddingX} y1={y} x2={width - paddingX} y2={y} stroke="var(--chart-grid)" strokeDasharray="3 3" />
              <text x={paddingX - 10} y={y + 4} textAnchor="end" className="chart-axis-text" fill="var(--chart-text)">{v}</text>
            </g>
          );
        })}

        {data.map((d, i) => {
          const barH = (d.count * plotH) / maxVal;
          const x = paddingX + i * (barW + gap);
          const y = height - paddingY - barH;

          return (
            <g key={i}>
              <rect
                x={x}
                y={y}
                width={barW}
                height={barH}
                fill="var(--chart-secondary)"
                rx="4"
                className="chart-bar"
              />
              <text x={x + barW / 2} y={y - 6} textAnchor="middle" className="chart-value-text" fill="var(--chart-text-primary)">
                {d.count}
              </text>
              <text x={x + barW / 2} y={height - paddingY + 16} textAnchor="middle" className="chart-axis-text" fill="var(--chart-text)">
                {d.name.split(" ")[0]}
              </text>
            </g>
          );
        })}
      </svg>
    );
  }, [stats.hourlyGroups]);

  const cameraStats = useMemo(() => {
    const maxCount = Math.max(...stats.cameraList.map(c => c.count), 1);

    return (
      <div className="camera-stats-list">
        {stats.cameraList.map((cam, idx) => {
          const pct = (cam.count / maxCount) * 100;
          return (
            <div key={idx} className="camera-stat-row">
              <div className="camera-name">{cam.name}</div>
              <div className="camera-bar-wrapper">
                <div className="camera-bar-fill" style={{ width: `${pct}%` }} />
              </div>
              <div className="camera-count">{cam.count}件</div>
            </div>
          );
        })}
      </div>
    );
  }, [stats.cameraList]);

  return (
    <div className="analytics-view">
      <div className="analytics-header">
        <h2>📊 AI検知イベント統計分析</h2>
        <p>CCTV映像から検知した車両事故・接触疑いイベントの統計概要です。</p>
      </div>

      {/* 기간 필터 */}
      <div className="filter-pills" style={{ marginBottom: "24px" }}>
        <button className={filterDays === 9999 ? "active" : ""} onClick={() => setFilterDays(9999)}>全期間</button>
        <button className={filterDays === 7 ? "active" : ""} onClick={() => setFilterDays(7)}>1週間</button>
        <button className={filterDays === 14 ? "active" : ""} onClick={() => setFilterDays(14)}>2週間</button>
        <button className={filterDays === 30 ? "active" : ""} onClick={() => setFilterDays(30)}>1か月</button>
        <button className={filterDays === 90 ? "active" : ""} onClick={() => setFilterDays(90)}>3か月</button>
      </div>

      {stats.totalEvents === 0 ? (
        <div className="empty-state" style={{ padding: "80px 20px" }}>選択した期間のイベント統計データはありません。</div>
      ) : (
        <>
          <div className="analytics-summary-cards">
            <div className="summary-card">
              <div className="card-icon">🎥</div>
              <div className="card-data">
                <span className="card-label">分析した映像</span>
                <span className="card-value">{stats.totalVideos}本</span>
              </div>
            </div>
            <div className="summary-card">
              <div className="card-icon">🚨</div>
              <div className="card-data">
                <span className="card-label">累計検知イベント</span>
                <span className="card-value">{stats.totalEvents}件</span>
              </div>
            </div>
            <div className="summary-card warning">
              <div className="card-icon">⚠️</div>
              <div className="card-data">
                <span className="card-label">確認が必要なイベント</span>
                <span className="card-value">{stats.pendingCount}件</span>
              </div>
            </div>
          </div>

          <div className="analytics-grid">
            <div className="analytics-chart-card">
              <h3> 日付別の検知イベント推移</h3>
              <div className="chart-wrapper">{dailyTrendChart}</div>
            </div>

            <div className="analytics-chart-card">
              <h3> イベント種別の割合</h3>
              <div className="chart-wrapper donut-wrapper">{donutChart}</div>
            </div>

            <div className="analytics-chart-card">
              <h3> 時間帯別の発生頻度</h3>
              <div className="chart-wrapper">{hourlyChartSvg}</div>
            </div>

            <div className="analytics-chart-card">
              <h3> カメラ別の検知状況</h3>
              <div className="chart-wrapper">{cameraStats}</div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// 대시보드 컴포넌트
function Dashboard({ onLogout, view, currentUser, onUpdateUser }) {
  const navigate = useNavigate();
  const { videoId } = useParams();
  const currentView = view === "analytics" ? "analytics" : "dashboard";

  // 상태 관리
  const [videos, setVideos] = useState([]); // API에서 로드한 영상 목록
  const [filterDays, setFilterDays] = useState(9999); // 기본 전체
  const [searchQuery, setSearchQuery] = useState(""); // 상단 검색어

  const [selectedVideo, setSelectedVideo] = useState(null); // null이면 홈(그리드) 화면, 값이 있으면 영상 재생 화면
  const [currentEventId, setCurrentEventId] = useState(null); // 현재 선택된 이벤트 마커
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0); // 실제 영상 재생 위치(초)
  const [selectedCalendarDate, setSelectedCalendarDate] = useState(null); // 달력에서 선택한 날짜
  const [datePickerVideos, setDatePickerVideos] = useState(null); // 동일 날짜 영상 복수 존재 시 선택 모달 데이터
  const [playbackSpeed, setPlaybackSpeed] = useState("1");
  const [volume, setVolume] = useState(70);
  const [quality, setQuality] = useState("auto");
  const [isTheaterMode, setIsTheaterMode] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [calendarMonth, setCalendarMonth] = useState(new Date("2026-05-01T00:00:00"));
  const playerContainerRef = useRef(null);
  const detectAbortControllerRef = useRef(null);
  const analyzeAbortControllerRef = useRef(null);

  // 바운딩박스 / 차량 지정 모드 상태 ("none" | "manual" | "auto")
  const [selectionMode, setSelectionMode] = useState("none");
  const [isBBoxMode, setIsBBoxMode] = useState(false);

  const [bboxList, setBboxList] = useState([]);
  const [currentDraw, setCurrentDraw] = useState(null); // { startX, startY, endX, endY }
  const [isDrawing, setIsDrawing] = useState(false);
  const bboxOverlayRef = useRef(null);
  const videoElRef = useRef(null); // 실제 <video> 엘리먼트 (원본 해상도 환산용)
  const [showBBoxPanel, setShowBBoxPanel] = useState(false);

  // YOLO 차량 자동 탐지 및 마우스 Hover 관련 상태
  const [isDetectingVehicles, setIsDetectingVehicles] = useState(false);
  const [detectedBoxes, setDetectedBoxes] = useState([]); // [{ id, class_name, confidence, bbox: [x1,y1,x2,y2] }]
  const [hoveredDetectedBox, setHoveredDetectedBox] = useState(null);


  // 사고감지 실행 / 분석 관련 상태
  const [showDetectConfirm, setShowDetectConfirm] = useState(false); // 확인 팝오버
  const [toast, setToast] = useState(null); // { type, message }
  // 동시 분석 작업 목록: [{ taskId, videoId, dateLabel, cameraLabel, estimatedSec, startedAt }]
  const [analyzingJobs, setAnalyzingJobs] = useState([]);
  const [showJobsDropdown, setShowJobsDropdown] = useState(false);
  const [now, setNow] = useState(() => Date.now()); // 진행률 계산용 현재시각(0.5s마다 갱신)
  const selectedVideoIdRef = useRef(null); // 폴링 중 최신 선택 영상 추적(stale closure 방지)
  const [clipEvent, setClipEvent] = useState(null); // CAM 클립 팝업 대상 이벤트

  // 업로드 모달
  const [showUpload, setShowUpload] = useState(false);

  // 영상 삭제 모드
  const [deleteMode, setDeleteMode] = useState(false);
  const [selectedForDelete, setSelectedForDelete] = useState([]); // 선택된 영상 id 목록

  const toggleDeleteSelect = (videoId) => {
    setSelectedForDelete((prev) =>
      prev.includes(videoId)
        ? prev.filter((id) => id !== videoId)
        : [...prev, videoId]
    );
  };

  const exitDeleteMode = () => {
    setDeleteMode(false);
    setSelectedForDelete([]);
  };

  const handleDeleteSelected = async () => {
    if (selectedForDelete.length === 0) {
      showToast("削除する映像を選択してください。", "warning");
      return;
    }
    if (!window.confirm(`選択した${selectedForDelete.length}本の映像を削除しますか？元に戻せません。`)) {
      return;
    }
    try {
      await Promise.all(selectedForDelete.map((id) => api.deleteVideo(id)));
      showToast(`${selectedForDelete.length}本の映像を削除しました。`, "success");
      exitDeleteMode();
      await loadVideos();
    } catch (e) {
      showToast(`削除に失敗しました：${e.message}`, "error");
    }
  };

  const showToast = (message, type = "warning") => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 2600);
  };

  // 프로필 & 설정 관련 상태
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [activeSettingsTab, setActiveSettingsTab] = useState("account"); // "account", "security"

  // 설정 정보 (현재 로그인 사용자 정보와 동기화)
  const currentUsername = currentUser?.username || "user";
  const [adminRealName, setAdminRealName] = useState(currentUser?.name || "");
  const [adminPhone, setAdminPhone] = useState(currentUser?.phone || "");
  const [adminEmail, setAdminEmail] = useState(currentUser?.email || "");

  useEffect(() => {
    if (currentUser) {
      setAdminRealName(currentUser.name || "");
      setAdminPhone(currentUser.phone || "");
      setAdminEmail(currentUser.email || "");
    }
  }, [currentUser]);

  const displayName = currentUser?.name || currentUser?.username || "ユーザー";
  const avatarInitial = (displayName[0] || currentUsername[0] || "U").toUpperCase();
  const roleLabel = currentUser?.role === "ADMIN" ? "システム管理者" : "一般ユーザー";

  // 비밀번호 변경 필드
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  // 테마 설정 (다크 모드)
  const [isDarkMode, setIsDarkMode] = useState(false);

  // 다크 모드 활성화 / 비활성화 제어
  useEffect(() => {
    if (isDarkMode) {
      document.body.classList.add("dark-mode");
    } else {
      document.body.classList.remove("dark-mode");
    }
  }, [isDarkMode]);

  // API에서 영상 목록 로드
  const loadVideos = async () => {
    try {
      const list = await api.listVideos(9999);
      setVideos(list);
      return list;
    } catch (e) {
      showToast(`映像一覧を読み込めませんでした：${e.message}`, "error");
      return [];
    }
  };

  useEffect(() => {
    // 마운트 시 영상 목록 비동기 로드 (setState는 await 이후 발생)
    // eslint-disable-next-line react-hooks/exhaustive-deps, react-hooks/set-state-in-effect
    loadVideos();
  }, []);

  // 라우트(videoId)에 맞춰 selectedVideo 설정
  useEffect(() => {
    if (view !== "watch" || !videoId) {
      // watch 화면이 아니면 선택 영상 해제 (라우트 동기화 가드)
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setSelectedVideo(null);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const v = await api.getVideo(videoId);
        if (cancelled) return;
        setSelectedVideo(v);
        setCurrentEventId(null);
        setIsPlaying(false);
        setIsTheaterMode(false);
        setIsSidebarOpen(true);
        setBboxList([]);
        setIsBBoxMode(false);
        setSelectionMode("none");
        setShowDetectConfirm(false);

        setCurrentTime(0);
        if (v.detected_vehicles) {
          try {
            setDetectedBoxes(JSON.parse(v.detected_vehicles));
          } catch {
            setDetectedBoxes([]);
          }
        } else {
          setDetectedBoxes([]);
        }
        setHoveredDetectedBox(null);

        if (v.date) {
          setCalendarMonth(new Date(`${v.date}T00:00:00`));
          setSelectedCalendarDate(v.date);
        }
      } catch (e) {
        showToast(`映像を読み込めませんでした：${e.message}`, "error");
        navigate("/videos");
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, videoId]);

  // 날짜 및 검색어 필터링 계산
  const filteredVideos = useMemo(() => {
    const today = new Date();
    const startDate = formatDate(addDays(today, -filterDays));
    let list =
      filterDays >= 9999
        ? videos
        : videos.filter((video) => video.date >= startDate);

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      list = list.filter(
        (v) =>
          (v.date || "").toLowerCase().includes(q) ||
          (v.camera || "").toLowerCase().includes(q) ||
          (v.events && v.events.some((e) => (e.title || "").toLowerCase().includes(q)))
      );
    }

    // 녹화 일자 최신순 정렬 (같은 날짜는 시작시간 → id 순)
    return [...list].sort((a, b) => {
      if (a.date !== b.date) return (b.date || "").localeCompare(a.date || "");
      if ((a.startTime || "") !== (b.startTime || ""))
        return (b.startTime || "").localeCompare(a.startTime || "");
      return b.id - a.id;
    });
  }, [filterDays, videos, searchQuery]);

  // 메인화면 영상 목록 페이지네이션 (6개 초과 시 페이지 분할)
  const VIDEOS_PER_PAGE = 6;
  const [currentPage, setCurrentPage] = useState(1);

  const totalPages = Math.max(1, Math.ceil(filteredVideos.length / VIDEOS_PER_PAGE));

  // 필터 조건(기간, 검색어) 변경 시 1페이지로 리셋
  useEffect(() => {
    setCurrentPage(1);
  }, [filterDays, searchQuery]);

  // 영상 삭제 등으로 인해 현재 페이지가 전체 페이지 수보다 커지면 조정
  useEffect(() => {
    if (currentPage > totalPages) {
      setCurrentPage(totalPages);
    }
  }, [totalPages, currentPage]);

  const paginatedVideos = useMemo(() => {
    const startIndex = (currentPage - 1) * VIDEOS_PER_PAGE;
    return filteredVideos.slice(startIndex, startIndex + VIDEOS_PER_PAGE);
  }, [filteredVideos, currentPage]);

  const handlePageChange = (newPage) => {
    setCurrentPage(newPage);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const getPageNumbers = () => {
    const maxButtons = 5;
    let start = Math.max(1, currentPage - 2);
    let end = Math.min(totalPages, start + maxButtons - 1);
    if (end - start + 1 < maxButtons) {
      start = Math.max(1, end - maxButtons + 1);
    }
    const pages = [];
    for (let i = start; i <= end; i++) {
      pages.push(i);
    }
    return pages;
  };

  const videosByDate = useMemo(() => {
    return getVideosByDateApi(videos);
  }, [videos]);

  const calendarDays = useMemo(() => getCalendarDays(calendarMonth), [calendarMonth]);
  const nextVideos = useMemo(() => {
    if (!selectedVideo || videos.length === 0) return [];
    const currentIndex = videos.findIndex((video) => video.id === selectedVideo.id);
    if (currentIndex === -1) return [];
    return Array.from({ length: Math.min(4, videos.length - 1) }, (_, index) => {
      return videos[(currentIndex + index + 1) % videos.length];
    });
  }, [selectedVideo, videos]);

  const moveCalendarMonth = (offset) => {
    setCalendarMonth((month) => new Date(month.getFullYear(), month.getMonth() + offset, 1));
  };

  const handleCalendarDateClick = (date) => {
    const dateText = formatDate(date);
    setSelectedCalendarDate(dateText); // 영상 유무와 무관하게 날짜 선택 표시
    const dayVideos = videosByDate[dateText] ?? [];
    if (dayVideos.length === 1) {
      handleWatchVideo(dayVideos[0]);
    } else if (dayVideos.length > 1) {
      setDatePickerVideos({ dateText, videos: dayVideos });
    }
  };

  // 영상을 클릭하여 시청 모드로 진입 → URL 이동
  const handleWatchVideo = (video) => {
    navigate(`/videos/${video.id}`);
  };

  // 홈(목록)으로 돌아가기
  const handleBackToHome = () => {
    navigate("/videos");
  };

  // 특정 사고 이벤트 1건 삭제
  const handleDeleteEvent = async (eventId, e) => {
    if (e) e.stopPropagation();
    if (!selectedVideo) return;
    try {
      await api.deleteEvent(selectedVideo.id, eventId);
      setSelectedVideo((prev) =>
        prev
          ? {
            ...prev,
            events: prev.events.filter((ev) => ev.id !== eventId),
          }
          : null
      );
      setVideos((prev) =>
        prev.map((v) =>
          v.id === selectedVideo.id
            ? { ...v, events: v.events.filter((ev) => ev.id !== eventId) }
            : v
        )
      );
      if (currentEventId === eventId) {
        setCurrentEventId(null);
      }
      showToast("検知イベントを削除しました。", "info");
    } catch (err) {
      showToast(`イベントの削除に失敗しました：${err.message}`, "error");
    }
  };

  // 영상의 모든 사고 이벤트 전체 삭제
  const handleClearAllEvents = async () => {
    if (!selectedVideo || selectedVideo.events.length === 0) return;
    if (!window.confirm("この映像で検知されたすべての事故イベントを削除しますか？")) return;
    try {
      await api.clearEvents(selectedVideo.id);
      setSelectedVideo((prev) => (prev ? { ...prev, events: [] } : null));
      setVideos((prev) =>
        prev.map((v) => (v.id === selectedVideo.id ? { ...v, events: [] } : v))
      );
      setCurrentEventId(null);
      showToast("すべての検知イベントを削除しました。", "info");
    } catch (err) {
      showToast(`イベント全体の削除に失敗しました：${err.message}`, "error");
    }
  };


  // 수동 차량 지정 모드 시작
  const handleStartManualDesignate = () => {
    setSelectionMode("manual");
    setIsBBoxMode(true);
    setHoveredDetectedBox(null);
    setShowBBoxPanel(true);
    showToast("🖱️ 手動車両指定モード：マウスでドラッグして事故車両を指定してください。", "info");
  };

  // 탐지/지정 모드 해제
  const handleEndDesignateMode = () => {
    setSelectionMode("none");
    setIsBBoxMode(false);
    setCurrentDraw(null);
    setIsDrawing(false);
    setHoveredDetectedBox(null);
  };

  // YOLO 차량 탐지 취소
  const handleCancelDetectVehicles = () => {
    if (detectAbortControllerRef.current) {
      detectAbortControllerRef.current.abort();
      detectAbortControllerRef.current = null;
    }
    setIsDetectingVehicles(false);
    setSelectionMode("none");
    showToast("車両検知をキャンセルしました。", "info");
  };

  // YOLO 차량 탐지 실행 (영상 pause → 백엔드 탐지 → Hover 선택 활성화)
  const handleAutoDetectVehicles = async () => {
    if (!selectedVideo) return;
    if (videoElRef.current) {
      videoElRef.current.pause();
      setIsPlaying(false);
    }
    setSelectionMode("auto");
    setIsDetectingVehicles(true);
    showToast("YOLOモデルで車両を検知しています…", "info");

    const controller = new AbortController();
    detectAbortControllerRef.current = controller;

    try {
      const res = await api.detectVehicles(selectedVideo.id, currentTime, controller.signal);
      const list = res.detected_vehicles || [];
      setDetectedBoxes(list);
      setIsBBoxMode(true);
      setShowBBoxPanel(true);
      if (list.length === 0) {
        showToast("検知された車両がありません。手動車両指定を利用してください。", "warning");
      } else {
        showToast(`${list.length}台の車両を検知しました。マウスを乗せると車両が表示され、クリックで選択できます。`, "success");
      }
    } catch (err) {
      if (err.name === "AbortError") {
        return;
      }
      showToast(`車両検知に失敗しました：${err.message}`, "error");
    } finally {
      setIsDetectingVehicles(false);
      detectAbortControllerRef.current = null;
    }
  };



  // '사고감지 실행' 클릭 → bbox 없으면 경고, 있으면 확인 팝오버
  const handleDetectClick = () => {
    if (bboxList.length === 0) {
      showToast("事故を検知する車両をドラッグで指定してください。", "warning");
      return;
    }
    setShowDetectConfirm(true);
  };

  // 사고감지 분석 작업 취소
  const handleCancelAnalysis = async (taskId) => {
    if (analyzeAbortControllerRef.current) {
      analyzeAbortControllerRef.current.abort();
      analyzeAbortControllerRef.current = null;
    }
    if (taskId) {
      try {
        await api.cancelTask(taskId);
        removeJob(taskId);
        showToast("事故検知分析をキャンセルしました。", "info");
        loadVideos();
      } catch (e) {
        showToast(`分析のキャンセルに失敗しました：${e.message}`, "error");
      }
    } else {
      showToast("分析リクエストをキャンセルしました。", "info");
    }
  };

  // 분석 태스크 상태 폴링 (analyzedId = 분석을 시작한 영상 id)
  const removeJob = (taskId) =>
    setAnalyzingJobs((prev) => prev.filter((j) => j.taskId !== taskId));

  const pollTask = (taskId, analyzedId) => {
    const poll = async () => {
      try {
        const status = await api.taskStatus(taskId);
        if (status.status === "SUCCESS") {
          removeJob(taskId);
          showToast(`分析完了：事故の疑いがある区間${status.events.length}件`, "success");
          // 분석한 영상을 아직 보고 있을 때만 화면 갱신 (다른 영상으로 이동했으면 덮어쓰지 않음)
          if (selectedVideoIdRef.current === analyzedId) {
            const v = await api.getVideo(analyzedId);
            setSelectedVideo(v);
          }
          loadVideos();
          return;
        }
        if (status.status === "FAILURE") {
          removeJob(taskId);
          showToast(`分析に失敗しました：${status.error_message || "エラー"}`, "error");
          return;
        }
        if (status.status === "CANCELLED") {
          removeJob(taskId);
          showToast("事故検知分析をキャンセルしました。", "info");
          return;
        }
        // 백엔드가 실제 추론 진행도(%)를 주면 해당 job에 반영
        if (typeof status.progress === "number") {
          setAnalyzingJobs((prev) =>
            prev.map((j) =>
              j.taskId === taskId ? { ...j, progress: status.progress } : j));
        }
        setTimeout(poll, 2000); // PENDING/PROCESSING → 재시도
      } catch (e) {
        removeJob(taskId);
        showToast(`ステータスの取得に失敗しました：${e.message}`, "error");
      }
    };
    poll();
  };

  // 확인 팝오버에서 '실행' → bbox 원본해상도 환산 후 분석 요청
  const runDetection = async () => {
    setShowDetectConfirm(false);
    const box = bboxList[0];
    if (!box) return;

    // 화면(오버레이) 좌표 → 원본 영상 해상도 픽셀 좌표 환산
    // bboxList의 좌표는 오버레이 div 기준(레터박스 여백 포함)이므로,
    // 그리기/선택과 동일한 레터박스 지오메트리로 되돌린다.
    const geom = getVideoGeom();
    const scale = geom ? geom.scale : 1;
    const offX = geom ? geom.offsetX : 0;
    const offY = geom ? geom.offsetY : 0;
    const bbox = {
      bbox_xmin: Math.round((box.xmin - offX) / scale),
      bbox_ymin: Math.round((box.ymin - offY) / scale),
      bbox_xmax: Math.round((box.xmax - offX) / scale),
      bbox_ymax: Math.round((box.ymax - offY) / scale),
    };

    // 분석 시작: 지정 모드를 끄고 해당 영상을 '분석 중'으로 표시
    const analyzedId = selectedVideo.id;
    setIsBBoxMode(false);
    setCurrentDraw(null);
    setIsDrawing(false);

    // 예상 소요시간(초) 추정: 슬라이딩 윈도우 수(총프레임/stride) 기반 (CPU ~1s/윈도우)
    // ※ 분모는 백엔드 config.PREDICT_WINDOW_STRIDE(현재 5)와 반드시 일치시킬 것
    const totalFrames = Math.round(
      (selectedVideo.duration || 0) * (selectedVideo.fps || 30));
    const estimatedSec = Math.max(10, Math.round((totalFrames / 5) * 1.1));
    const job = {
      taskId: null,
      videoId: analyzedId,
      dateLabel: selectedVideo.date,
      cameraLabel: selectedVideo.camera,
      estimatedSec,
      startedAt: Date.now(),
    };

    const controller = new AbortController();
    analyzeAbortControllerRef.current = controller;

    try {
      const res = await api.analyze(analyzedId, bbox, controller.signal);
      job.taskId = res.task_id;
      setAnalyzingJobs((prev) => [...prev, job]);
      pollTask(res.task_id, analyzedId);
    } catch (e) {
      if (e.name === "AbortError") {
        return;
      }
      showToast(`分析リクエストに失敗しました：${e.message}`, "error");
    } finally {
      analyzeAbortControllerRef.current = null;
    }
  };

  const handleToggleFullscreen = () => {
    const player = playerContainerRef.current;
    if (!player) return;

    if (document.fullscreenElement) {
      document.exitFullscreen();
      return;
    }

    player.requestFullscreen?.();
  };

  // ── 실제 <video> 재생 제어 ──────────────────────────────
  const togglePlay = () => {
    const v = videoElRef.current;
    if (!v) return;
    if (v.paused) v.play();
    else v.pause();
  };

  // 재생바 클릭으로 탐색(seek)
  const handleSeek = (e) => {
    const v = videoElRef.current;
    if (!v || !v.duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    v.currentTime = ratio * v.duration;
  };

  // 이벤트 마커 클릭 → 해당 시점으로 이동 후 재생
  const seekToEvent = (event) => {
    setCurrentEventId(event.id);
    const v = videoElRef.current;
    if (v) {
      v.currentTime = event.timestamp;
      v.play();
    }
  };

  // 배속/볼륨을 실제 video에 반영
  useEffect(() => {
    const v = videoElRef.current;
    if (v) v.playbackRate = parseFloat(playbackSpeed);
  }, [playbackSpeed, selectedVideo]);

  useEffect(() => {
    const v = videoElRef.current;
    if (v) v.volume = volume / 100;
  }, [volume, selectedVideo]);

  // 폴링 콜백에서 현재 보고 있는 영상 id를 최신으로 참조 (stale closure 방지)
  useEffect(() => {
    selectedVideoIdRef.current = selectedVideo?.id ?? null;
  }, [selectedVideo]);

  // 현재 보고 있는 영상이 분석 중인지 (영상별로 판정 — 다른 영상엔 영향 없음)
  const isAnalyzing =
    !!selectedVideo && analyzingJobs.some((j) => j.videoId === selectedVideo.id);

  // 분석 작업이 있는 동안 진행률 바를 0.5초마다 갱신
  useEffect(() => {
    if (analyzingJobs.length === 0) return;
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, [analyzingJobs.length]);

  // 작업별 진행률(%) / 남은 시간(초) 계산 (시간 기반 추정 — 완료 전까지 96%에서 대기)
  const jobProgress = (job) => {
    const elapsed = (now - job.startedAt) / 1000;
    // 백엔드가 실제 추론 진행도(%)를 보고했으면 그걸 우선 사용
    if (typeof job.progress === "number") {
      const pct = Math.min(99, job.progress);
      // 남은 시간: 현재까지 경과/진행률로 역산 (진행률 데이터 기반)
      const remain = job.progress > 0
        ? Math.max(0, Math.ceil((elapsed * (100 - job.progress)) / job.progress))
        : Math.ceil(job.estimatedSec);
      return { pct, remain };
    }
    // 폴백: 아직 진행도 보고 전(PENDING 등)이면 시간 기반 추정
    const pct = Math.min(96, (elapsed / job.estimatedSec) * 100);
    const remain = Math.max(0, Math.ceil(job.estimatedSec - elapsed));
    return { pct, remain };
  };

  // 오버레이 div 안에서 실제 영상 콘텐츠가 차지하는 영역(object-fit:contain 레터박스)과
  // 원본해상도→화면 스케일을 반환. bbox 그리기/클릭 좌표 환산을 이걸로 통일한다.
  const getVideoGeom = () => {
    const el = bboxOverlayRef.current;
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    const vid = videoElRef.current;
    const vw = (vid && vid.videoWidth) || selectedVideo?.width || rect.width;
    const vh = (vid && vid.videoHeight) || selectedVideo?.height || rect.height;
    const scale = Math.min(rect.width / vw, rect.height / vh); // contain
    return {
      rect,
      scale,
      offsetX: (rect.width - vw * scale) / 2, // 좌우 레터박스 여백
      offsetY: (rect.height - vh * scale) / 2, // 상하 레터박스 여백
    };
  };

  return (
    <div className="dashboard-layout">
      {/* 상단 네비게이션 바 */}
      <header className="top-navbar">
        <div className="nav-section nav-left">
          <button className="nav-logo nav-home-btn" onClick={handleBackToHome} aria-label="SIOTホームへ移動">
            <img src="/siot-logo.svg" alt="SIOT" />
          </button>
          <div className="nav-search-bar">
            <span className="search-icon">🔍</span>
            <input
              type="text"
              placeholder="日付、カメラ、イベントを検索…"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            {searchQuery && (
              <button className="search-clear-btn" onClick={() => setSearchQuery("")}>
                ✕
              </button>
            )}
          </div>
        </div>

        <div className="nav-section nav-center">
          {analyzingJobs.length === 0 ? (
            <div className="analysis-status-box analysis-status-idle">
              <span className="analysis-idle-text">現在分析中の映像はありません</span>
            </div>
          ) : (() => {
            const lead = analyzingJobs[0];
            const { pct, remain } = jobProgress(lead);
            return (
              <div className="analysis-status-box">
                <div className="analysis-status-main">
                  <div className="analysis-status-info">
                    <span className="analysis-status-label">
                      🔍 {lead.dateLabel} 分析中
                    </span>
                    <span className="analysis-status-eta">残り約{remain}秒</span>
                  </div>
                  <div className="analysis-progress-track">
                    <div
                      className="analysis-progress-fill"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>

                <button
                  className="analysis-jobs-toggle"
                  onClick={() => setShowJobsDropdown((v) => !v)}
                  title="분석 중인 영상 목록"
                >
                  {analyzingJobs.length}件 ▾
                </button>

                {showJobsDropdown && (
                  <>
                    <div
                      className="dropdown-overlay"
                      onClick={() => setShowJobsDropdown(false)}
                    />
                    <div className="analysis-jobs-dropdown">
                      <div className="analysis-jobs-dropdown-title">
                        分析中の映像（{analyzingJobs.length}件）
                      </div>
                      {analyzingJobs.map((job) => {
                        const p = jobProgress(job);
                        return (
                          <div className="analysis-job-row" key={job.taskId ?? job.startedAt}>
                            <div className="analysis-job-top">
                              <span className="analysis-job-name">
                                {job.dateLabel} · {job.cameraLabel}
                              </span>
                              <div className="analysis-job-top-right">
                                <span className="analysis-job-pct">{Math.round(p.pct)}%</span>
                                <button
                                  className="analysis-job-cancel-btn"
                                  onClick={() => handleCancelAnalysis(job.taskId)}
                                  title="분석 취소"
                                >
                                  🛑 キャンセル
                                </button>
                              </div>
                            </div>
                            <div className="analysis-progress-track">
                              <div
                                className="analysis-progress-fill"
                                style={{ width: `${p.pct}%` }}
                              />
                            </div>
                            <span className="analysis-job-eta">残り約{p.remain}秒</span>
                          </div>
                        );
                      })}
                    </div>
                  </>
                )}
              </div>
            );
          })()}
        </div>
        <div className="nav-section nav-right">
          <div className="profile-menu-container">
            <button
              className="profile-trigger-btn simple-avatar-trigger"
              onClick={() => setIsProfileOpen(!isProfileOpen)}
              aria-label="プロフィールメニューを開く"
            >
              <div className="profile-avatar">
                <span>{avatarInitial}</span>
              </div>
            </button>

            {isProfileOpen && (
              <>
                <div className="dropdown-overlay" onClick={() => setIsProfileOpen(false)} />
                <div className="profile-dropdown-menu">
                  {/* 관리자/사용자 정보 요약 Header */}
                  <div className="dropdown-header">
                    <div className="header-avatar">{avatarInitial}</div>
                    <div className="header-info">
                      <span className="info-name">{displayName} ({currentUsername})</span>
                      <span className="info-role">{roleLabel}</span>
                    </div>
                  </div>

                  <div className="dropdown-divider" />

                  {/* 내 정보 설정 (Account Settings) */}
                  <div className="dropdown-section-title">個人情報設定</div>
                  <button
                    className="dropdown-item"
                    onClick={() => {
                      setIsProfileOpen(false);
                      setIsSettingsOpen(true);
                      setActiveSettingsTab("account");
                    }}
                  >
                    👤 {currentUser?.role === "ADMIN" ? "管理者情報を編集" : "個人情報を編集"}
                  </button>
                  <button
                    className="dropdown-item"
                    onClick={() => {
                      setIsProfileOpen(false);
                      setIsSettingsOpen(true);
                      setActiveSettingsTab("security");
                    }}
                  >
                    🔑 パスワード変更
                  </button>

                  <div className="dropdown-divider" />

                  {/* 화면 이동 */}
                  <div className="dropdown-section-title">画面移動</div>
                  <button
                    className={`dropdown-item ${currentView === "analytics" ? "active-menu-item" : ""}`}
                    onClick={() => {
                      setIsProfileOpen(false);
                      navigate("/analytics");
                    }}
                  >
                    📊 イベント統計グラフ
                  </button>

                  <div className="dropdown-divider" />

                  {/* 테마 설정 (Appearance) */}
                  <div className="dropdown-section-title">テーマ設定</div>
                  <div className="dropdown-item-toggle">
                    <span>🌙 ダークモード</span>
                    <label className="switch-mini">
                      <input
                        type="checkbox"
                        checked={isDarkMode}
                        onChange={(e) => setIsDarkMode(e.target.checked)}
                      />
                      <span className="slider-mini round"></span>
                    </label>
                  </div>

                  <div className="dropdown-divider" />

                  {/* 로그아웃 (Logout) */}
                  <button
                    className="dropdown-item logout-item"
                    onClick={() => {
                      setIsProfileOpen(false);
                      onLogout();
                    }}
                  >
                    🚪 ログアウト
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      </header>

      {/* 컨텐츠 영역: 통계 뷰 또는 모니터링 뷰 */}
      {currentView === "analytics" ? (
        <AnalyticsView
          filteredVideos={filteredVideos}
          filterDays={filterDays}
          setFilterDays={setFilterDays}
        />
      ) : !selectedVideo ? (
        <main className="home-view">
          {/* 기간 필터 및 영상 관리 툴바 */}
          <div className="home-toolbar">
            <div className="filter-pills">
              <button className={filterDays === 9999 ? "active" : ""} onClick={() => setFilterDays(9999)}>全期間</button>
              <button className={filterDays === 7 ? "active" : ""} onClick={() => setFilterDays(7)}>1週間</button>
              <button className={filterDays === 14 ? "active" : ""} onClick={() => setFilterDays(14)}>2週間</button>
              <button className={filterDays === 30 ? "active" : ""} onClick={() => setFilterDays(30)}>1か月</button>
              <button className={filterDays === 90 ? "active" : ""} onClick={() => setFilterDays(90)}>3か月</button>
            </div>

            <div className="home-toolbar-actions">
              {deleteMode ? (
                <>
                  <button
                    className="delete-select-all-btn"
                    onClick={() => {
                      if (selectedForDelete.length === filteredVideos.length && filteredVideos.length > 0) {
                        setSelectedForDelete([]);
                      } else {
                        setSelectedForDelete(filteredVideos.map((v) => v.id));
                      }
                    }}
                  >
                    {selectedForDelete.length === filteredVideos.length && filteredVideos.length > 0
                      ? "☑ すべて解除"
                      : "☐ すべて選択"}
                  </button>
                  <button className="delete-cancel-btn" onClick={exitDeleteMode}>
                    キャンセル
                  </button>
                  <button className="delete-confirm-btn" onClick={handleDeleteSelected}>
                    🗑 選択項目を削除 ({selectedForDelete.length})
                  </button>
                </>
              ) : (
                <>
                  <button className="toolbar-upload-btn" onClick={() => setShowUpload(true)}>
                    ⬆ 映像をアップロード
                  </button>
                  <button className="toolbar-delete-btn" onClick={() => setDeleteMode(true)}>
                    🗑 映像を削除
                  </button>
                </>
              )}
            </div>
          </div>


          {/* 영상 썸네일 그리드 */}
          <div className="video-grid">
            {filteredVideos.length === 0 ? (
              <div className="empty-state">選択した期間の映像はありません。</div>
            ) : (
              paginatedVideos.map((video) => {
                const isSelected = selectedForDelete.includes(video.id);
                return (
                  <div
                    key={video.id}
                    className={`video-card ${deleteMode ? "delete-mode" : ""} ${isSelected ? "delete-selected" : ""}`}
                    onClick={() =>
                      deleteMode ? toggleDeleteSelect(video.id) : handleWatchVideo(video)
                    }
                  >
                    <div className="video-thumbnail">
                      {/* 영상 첫 프레임 썸네일 */}
                      <img
                        className="video-thumbnail-img"
                        src={api.thumbnailUrl(video.id)}
                        alt={`${video.date} ${video.camera} サムネイル`}
                        loading="lazy"
                        onError={(e) => { e.currentTarget.style.display = "none"; }}
                      />
                      <span className="event-count-badge">イベント {video.events.length}件</span>
                      {deleteMode && (
                        <span className={`delete-check ${isSelected ? "checked" : ""}`}>
                          {isSelected ? "✓" : ""}
                        </span>
                      )}
                    </div>
                    <div className="video-info">
                      <h3>{getVideoDisplayTitle(video, videos)}</h3>
                      <p>映像の長さ：{formatTime(video.duration)}</p>
                    </div>
                  </div>
                );
              })
            )}
          </div>

          {/* 페이지네이션 (영상 개수가 6개를 초과할 때 노출) */}
          {filteredVideos.length > VIDEOS_PER_PAGE && (
            <div className="pagination-bar">
              <button
                className="pagination-btn pagination-nav"
                onClick={() => handlePageChange(1)}
                disabled={currentPage === 1}
                title="最初のページ"
              >
                «
              </button>
              <button
                className="pagination-btn pagination-nav"
                onClick={() => handlePageChange(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                title="前のページ"
              >
                ‹
              </button>

              <div className="pagination-pages">
                {getPageNumbers().map((pageNum) => (
                  <button
                    key={pageNum}
                    className={`pagination-btn pagination-num ${currentPage === pageNum ? "active" : ""}`}
                    onClick={() => handlePageChange(pageNum)}
                  >
                    {pageNum}
                  </button>
                ))}
              </div>

              <button
                className="pagination-btn pagination-nav"
                onClick={() => handlePageChange(Math.min(totalPages, currentPage + 1))}
                disabled={currentPage === totalPages}
                title="次のページ"
              >
                ›
              </button>
              <button
                className="pagination-btn pagination-nav"
                onClick={() => handlePageChange(totalPages)}
                disabled={currentPage === totalPages}
                title="最後のページ"
              >
                »
              </button>

              <span className="pagination-info">
                {currentPage} / {totalPages}ページ（全{filteredVideos.length}件）
              </span>
            </div>
          )}
        </main>
      ) : (
        <main className={`watch-view ${isTheaterMode ? "theater-view" : ""}`}>
          <div className="watch-header">
            <button className="back-btn" onClick={handleBackToHome}>
              ← 一覧に戻る
            </button>

            {/* 영상(player) 컬럼에 맞춰 정렬되는 오른쪽 영역 */}
            <div className="watch-header-right">
              {/* 탐지하기 + 수동 차량 지정 + 사고감지 실행 버튼 그룹 (나란히) */}
              <div className="watch-header-actions">
                <button
                  className={`auto-detect-btn ${selectionMode === "auto" ? "active" : ""} ${isDetectingVehicles ? "detecting" : ""}`}
                  disabled={isAnalyzing}
                  onClick={isDetectingVehicles ? handleCancelDetectVehicles : handleAutoDetectVehicles}
                  title={isDetectingVehicles ? "実行中の車両検知をキャンセル" : "YOLOで現在のフレームから車両を自動検知して選択"}
                >
                  {isDetectingVehicles ? "🛑 検知をキャンセル" : "🔍 車両を検知"}
                </button>

                <button
                  className={`bbox-designate-btn ${selectionMode === "manual" ? "active" : ""}`}
                  disabled={isAnalyzing}
                  onClick={() => {
                    if (selectionMode === "manual") {
                      handleEndDesignateMode();
                    } else {
                      handleStartManualDesignate();
                    }
                  }}
                  title={isAnalyzing ? "分析中は指定できません" : (selectionMode === "manual" ? "手動指定モードを終了" : "ドラッグして車両を指定")}
                >
                  {selectionMode === "manual" ? "手動指定を終了" : "車両を手動指定"}
                </button>

                {selectionMode !== "none" && (
                  <button
                    className="mode-exit-btn"
                    onClick={handleEndDesignateMode}
                    title="すべての指定モードを解除"
                  >
                    ✕ モードを解除
                  </button>
                )}



                {/* 사고감지 실행 버튼 + 확인 팝오버 / 취소 버튼 */}
                <div className="detect-btn-wrapper">
                  {isAnalyzing ? (
                    <button
                      className="detect-run-btn analyzing-cancel-btn"
                      onClick={() => {
                        const currentJob = analyzingJobs.find((j) => j.videoId === selectedVideo?.id);
                        handleCancelAnalysis(currentJob?.taskId);
                      }}
                      title="実行中の事故検知分析をキャンセル"
                    >
                      🛑 事故検知をキャンセル
                    </button>
                  ) : (
                    <button
                      className="detect-run-btn"
                      onClick={handleDetectClick}
                      title="選択した車両の事故予測区間を検知"
                    >
                      事故検知を実行
                    </button>
                  )}

                  {showDetectConfirm && (
                    <>
                      <div
                        className="detect-popover-overlay"
                        onClick={() => setShowDetectConfirm(false)}
                      />
                      <div className="detect-popover">
                        <p className="detect-popover-text">
                          選択した車両の事故予測区間を検知しますか？
                        </p>
                        <div className="detect-popover-actions">
                          <button className="detect-popover-run" onClick={runDetection}>
                            実行
                          </button>
                          <button
                            className="detect-popover-cancel"
                            onClick={() => setShowDetectConfirm(false)}
                          >
                            戻る
                          </button>
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </div>

              <div className="watch-header-metadata">
                <span className="metadata-item"><strong>カメラ：</strong> {selectedVideo.camera}</span>
                <span className="metadata-item"><strong>録画映像：</strong> {getVideoDisplayTitle(selectedVideo, videos)} ({selectedVideo.startTime})</span>
                <span className="metadata-item"><strong>イベント合計：</strong> {selectedVideo.events.length}件検知</span>
              </div>
            </div>
          </div>

          <div className={`watch-layout ${isTheaterMode ? "theater-mode" : ""} ${!isSidebarOpen ? "sidebar-collapsed" : ""}`}>
            <aside className={`event-sidebar ${!isSidebarOpen ? "collapsed" : ""}`}>
              {isSidebarOpen ? (
                <>
                  <div className="event-sidebar-header">
                    <div>
                      <h3>検知イベント一覧</h3>
                      <p className="event-summary">合計{selectedVideo.events.length}件のイベントがあります。</p>
                    </div>
                    <div className="event-header-actions">
                      {selectedVideo.events.length > 0 && (
                        <button
                          className="event-clear-all-btn"
                          onClick={handleClearAllEvents}
                          title="この映像のすべての検知イベントを削除"
                        >
                          🗑 すべて削除
                        </button>
                      )}
                      <button
                        className="sidebar-icon-btn"
                        onClick={() => setIsSidebarOpen(false)}
                        aria-label="이벤트 목록 닫기"
                      >
                        ×
                      </button>
                    </div>
                  </div>

                  <div className="event-list">
                    {selectedVideo.events.length === 0 ? (
                      <div className="event-empty-hint">検知されたイベントはありません。</div>
                    ) : (
                      selectedVideo.events.map((event) => (
                        <div
                          key={event.id}
                          className={`event-item ${currentEventId === event.id ? "active" : ""}`}
                          onClick={() => setCurrentEventId(event.id)}
                        >
                          <div className="event-item-top">
                            <span className="event-time-pill">{formatTime(event.timestamp)}</span>
                            <span className="event-actual-time-text">
                              {formatActualTime(selectedVideo.date, selectedVideo.startTime, event.timestamp)}
                            </span>
                            <div className="event-item-actions">
                              <button
                                className="event-play-icon-btn"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setCurrentEventId(event.id);
                                  if (event.hasClip) {
                                    setClipEvent(event);
                                  } else {
                                    showToast("このイベントのCAMクリップはまだありません。", "warning");
                                  }
                                }}
                                aria-label={`${formatTime(event.timestamp)} 事故区間CAMクリップを再生`}
                                title="事故区間CAMクリップを見る"
                              >
                                ▶
                              </button>
                              <button
                                className="event-delete-item-btn"
                                onClick={(e) => handleDeleteEvent(event.id, e)}
                                title="イベントを削除"
                              >
                                🗑
                              </button>
                            </div>
                          </div>
                          <div className="event-item-bottom">
                            <h4 className="event-title-text">{event.title}</h4>
                          </div>
                        </div>
                      ))
                    )}
                  </div>


                  {/* 달력을 사이드바 이벤트 목록 하단으로 삽입 */}
                  <div className="sidebar-calendar-container">
                    <section className="event-calendar-panel" style={{ background: "transparent", padding: 0 }}>
                      <div className="calendar-header">
                        <button className="calendar-nav-btn" onClick={() => moveCalendarMonth(-1)} aria-label="前月">
                          ‹
                        </button>
                        <h2>{formatMonthLabel(calendarMonth)}</h2>
                        <button className="calendar-nav-btn" onClick={() => moveCalendarMonth(1)} aria-label="翌月">
                          ›
                        </button>
                      </div>

                      <div className="calendar-weekdays">
                        {["日", "月", "火", "水", "木", "金", "土"].map((day) => (
                          <span key={day}>{day}</span>
                        ))}
                      </div>

                      <div className="calendar-grid">
                        {calendarDays.map((date, index) => {
                          if (!date) {
                            return <div key={`empty-${index}`} className="calendar-day empty" />;
                          }

                          const dateText = formatDate(date);
                          const videos = videosByDate[dateText] ?? [];
                          const hasVideo = videos.length > 0;
                          const isSelectedDate = (selectedCalendarDate ?? selectedVideo.date) === dateText;

                          // 영상의 총 이벤트 건수를 합산하여 강도를 계산
                          const totalEvents = videos.reduce((acc, v) => acc + v.events.length, 0);

                          let intensityClass = "";
                          if (totalEvents === 1) intensityClass = "intensity-1";
                          else if (totalEvents === 2) intensityClass = "intensity-2";
                          else if (totalEvents >= 3) intensityClass = "intensity-3";

                          return (
                            <button
                              key={dateText}
                              className={`calendar-day ${hasVideo ? `has-video ${intensityClass}` : ""} ${isSelectedDate ? "selected" : ""}`}
                              onClick={() => handleCalendarDateClick(date)}
                              title={hasVideo ? `映像${videos.length}本 / イベント${totalEvents}件` : "映像なし"}
                            >
                              <span className="calendar-date-number">{date.getDate()}</span>
                              {videos.length > 1 && (
                                <span className="calendar-multi-count-badge" title={`映像${videos.length}本`}>
                                  {videos.length}本
                                </span>
                              )}
                            </button>
                          );
                        })}
                      </div>
                    </section>
                  </div>
                </>
              ) : (
                <button
                  className="sidebar-open-btn"
                  onClick={() => setIsSidebarOpen(true)}
                >
                  イベント一覧を開く
                </button>
              )}

              {!isSidebarOpen && nextVideos.length > 0 && (
                <section className="next-video-section">
                  <h3>次の映像</h3>
                  <div className="next-video-list">
                    {nextVideos.map((video) => (
                      <button
                        key={video.id}
                        className="next-video-card"
                        onClick={() => handleWatchVideo(video)}
                      >
                        <strong>{video.date}</strong>
                        <span>{video.camera}</span>
                        <small>{video.startTime} · イベント {video.events.length}件 · {formatTime(video.duration)}</small>
                      </button>
                    ))}
                  </div>
                </section>
              )}
            </aside>

            {/* 우측/중앙: 메인 비디오 플레이어 */}
            <section className="player-section">
              {/* 지정 모드 안내문 (영상 상단을 가리지 않도록 비디오 플레이어 바깥 위에 배치) */}
              {selectionMode !== "none" && (
                <div className="bbox-mode-guide-bar">
                  <span className="bbox-guide-icon">{selectionMode === "auto" ? "🔍" : "🖱️"}</span>
                  <span>
                    {selectionMode === "auto"
                      ? "YOLO自動検知モード：マウスを乗せると検知車両が表示され、クリックで選択できます。"
                      : "手動車両指定モード：マウスでドラッグして事故車両を指定してください。"}
                  </span>
                </div>
              )}

              <div className="player-container" ref={playerContainerRef}>
                <div
                  className={`mock-video-player ${isBBoxMode ? "bbox-mode-active" : ""}`}
                  ref={bboxOverlayRef}
                  style={{
                    position: "relative",
                    userSelect: "none",
                    aspectRatio: selectedVideo?.width && selectedVideo?.height
                      ? `${selectedVideo.width} / ${selectedVideo.height}`
                      : "16 / 9",
                  }}
                  onMouseDown={(e) => {
                    if (!isBBoxMode) return;
                    if (selectionMode === "auto" && hoveredDetectedBox && getVideoGeom()) {
                      const { scale, offsetX, offsetY } = getVideoGeom();
                      const [x1, y1, x2, y2] = hoveredDetectedBox.bbox;
                      const dispX1 = Math.round(offsetX + x1 * scale);
                      const dispY1 = Math.round(offsetY + y1 * scale);
                      const dispX2 = Math.round(offsetX + x2 * scale);
                      const dispY2 = Math.round(offsetY + y2 * scale);

                      setBboxList([{ id: Date.now(), xmin: dispX1, ymin: dispY1, xmax: dispX2, ymax: dispY2 }]);
                      showToast(`${hoveredDetectedBox.class_name}（#${hoveredDetectedBox.id + 1}）の車両を選択しました。`, "success");
                      return;
                    }
                    if (selectionMode === "manual") {
                      const rect = bboxOverlayRef.current.getBoundingClientRect();
                      const x = Math.round(e.clientX - rect.left);
                      const y = Math.round(e.clientY - rect.top);
                      setIsDrawing(true);
                      setCurrentDraw({ startX: x, startY: y, endX: x, endY: y });
                    }
                  }}
                  onMouseMove={(e) => {
                    if (!isBBoxMode) return;
                    const rect = bboxOverlayRef.current.getBoundingClientRect();

                    if (selectionMode === "manual" && isDrawing) {
                      const x = Math.round(e.clientX - rect.left);
                      const y = Math.round(e.clientY - rect.top);
                      setCurrentDraw((prev) => (prev ? { ...prev, endX: x, endY: y } : null));
                      return;
                    }

                    if (selectionMode === "auto" && detectedBoxes.length > 0 && getVideoGeom()) {
                      const { scale, offsetX, offsetY } = getVideoGeom();
                      // 화면(오버레이) 좌표 → 레터박스 여백 제거 → 원본 해상도 좌표
                      const realX = (e.clientX - rect.left - offsetX) / scale;
                      const realY = (e.clientY - rect.top - offsetY) / scale;

                      const hit = detectedBoxes.find((box) => {
                        const [x1, y1, x2, y2] = box.bbox;
                        return realX >= x1 && realX <= x2 && realY >= y1 && realY <= y2;
                      });

                      setHoveredDetectedBox(hit || null);
                    } else if (selectionMode !== "auto") {
                      setHoveredDetectedBox(null);
                    }
                  }}
                  onMouseUp={(e) => {
                    if (selectionMode === "manual" && isDrawing && currentDraw) {
                      const rect = bboxOverlayRef.current.getBoundingClientRect();
                      const x = Math.round(e.clientX - rect.left);
                      const y = Math.round(e.clientY - rect.top);
                      const xmin = Math.min(currentDraw.startX, x);
                      const ymin = Math.min(currentDraw.startY, y);
                      const xmax = Math.max(currentDraw.startX, x);
                      const ymax = Math.max(currentDraw.startY, y);
                      if (xmax - xmin > 5 && ymax - ymin > 5) {
                        setBboxList([{ id: Date.now(), xmin, ymin, xmax, ymax }]);
                      }
                      setCurrentDraw(null);
                      setIsDrawing(false);
                    }
                  }}
                  onMouseLeave={() => {
                    if (selectionMode === "manual" && isDrawing && currentDraw) {
                      const xmin = Math.min(currentDraw.startX, currentDraw.endX);
                      const ymin = Math.min(currentDraw.startY, currentDraw.endY);
                      const xmax = Math.max(currentDraw.startX, currentDraw.endX);
                      const ymax = Math.max(currentDraw.startY, currentDraw.endY);
                      if (xmax - xmin > 5 && ymax - ymin > 5) {
                        setBboxList([{ id: Date.now(), xmin, ymin, xmax, ymax }]);
                      }
                    }
                    setCurrentDraw(null);
                    setIsDrawing(false);
                    setHoveredDetectedBox(null);
                  }}
                >

                  {/* 실제 업로드 영상 재생 (bbox 모드에서는 포인터 이벤트를 컨테이너로 전달) */}
                  <video
                    ref={videoElRef}
                    className="real-video"
                    src={api.streamUrl(selectedVideo.id)}
                    controls={false}

                    onPlay={() => setIsPlaying(true)}
                    onPause={() => setIsPlaying(false)}
                    onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
                    style={{
                      display: "block",
                      width: "100%",
                      height: "100%",
                      objectFit: "contain",
                      background: "#000",
                      pointerEvents: isBBoxMode ? "none" : "auto",
                    }}
                  />

                  {/* 저장된 바운딩 박스 렌더링 (단일 박스) */}
                  {bboxList.map((box) => (
                    <div
                      key={box.id}
                      className="bbox-drawn"
                      style={{
                        left: box.xmin,
                        top: box.ymin,
                        width: box.xmax - box.xmin,
                        height: box.ymax - box.ymin,
                      }}
                    >
                      {isBBoxMode && (
                        <button
                          className="bbox-delete-btn"
                          onClick={(e) => {
                            e.stopPropagation();
                            setBboxList((prev) => prev.filter((b) => b.id !== box.id));
                          }}
                          title="박스 삭제"
                        >
                          ✕
                        </button>
                      )}
                    </div>
                  ))}

                  {/* 현재 수동 드래그 중인 박스 미리보기 */}
                  {selectionMode === "manual" && isDrawing && currentDraw && (() => {
                    const xmin = Math.min(currentDraw.startX, currentDraw.endX);
                    const ymin = Math.min(currentDraw.startY, currentDraw.endY);
                    const xmax = Math.max(currentDraw.startX, currentDraw.endX);
                    const ymax = Math.max(currentDraw.startY, currentDraw.endY);
                    return (
                      <div
                        className="bbox-preview"
                        style={{ left: xmin, top: ymin, width: xmax - xmin, height: ymax - ymin }}
                      >
                        <span className="bbox-preview-label">
                          ({xmin}, {ymin}) → ({xmax}, {ymax})
                        </span>
                      </div>
                    );
                  })()}

                  {/* YOLO 자동 탐지 모드: 탐지된 모든 bbox를 항상 표시 (진단용 #번호·클래스·신뢰도) */}
                  {selectionMode === "auto" && detectedBoxes.length > 0 && getVideoGeom() && (() => {
                    const { scale, offsetX, offsetY } = getVideoGeom();
                    return detectedBoxes.map((box) => {
                      const [x1, y1, x2, y2] = box.bbox;
                      const dispX1 = Math.round(offsetX + x1 * scale);
                      const dispY1 = Math.round(offsetY + y1 * scale);
                      const dispX2 = Math.round(offsetX + x2 * scale);
                      const dispY2 = Math.round(offsetY + y2 * scale);
                      const isHovered = hoveredDetectedBox && hoveredDetectedBox.id === box.id;
                      return (
                        <div
                          key={box.id}
                          style={{
                            position: "absolute",
                            left: dispX1,
                            top: dispY1,
                            width: Math.max(4, dispX2 - dispX1),
                            height: Math.max(4, dispY2 - dispY1),
                            border: `2px solid ${isHovered ? "#16a34a" : "rgba(34,197,94,0.7)"}`,
                            boxSizing: "border-box",
                            borderRadius: 2,
                            pointerEvents: "none",
                          }}
                        >
                          <span
                            style={{
                              position: "absolute",
                              top: -17,
                              left: -1,
                              background: "rgba(22,163,74,0.9)",
                              color: "#fff",
                              fontSize: 11,
                              lineHeight: "15px",
                              padding: "0 4px",
                              borderRadius: 3,
                              whiteSpace: "nowrap",
                            }}
                          >
                            #{box.id + 1} {box.class_name} {Math.round((box.confidence || 0) * 100)}%
                          </span>
                        </div>
                      );
                    });
                  })()}

                  {/* YOLO 자동 탐지 모드: 마우스 Hover 미리보기 (Hover 시에만 가상 노출) */}
                  {selectionMode === "auto" && hoveredDetectedBox && getVideoGeom() && (() => {
                    const { scale, offsetX, offsetY } = getVideoGeom();
                    const [x1, y1, x2, y2] = hoveredDetectedBox.bbox;
                    const dispX1 = Math.round(offsetX + x1 * scale);
                    const dispY1 = Math.round(offsetY + y1 * scale);
                    const dispX2 = Math.round(offsetX + x2 * scale);
                    const dispY2 = Math.round(offsetY + y2 * scale);

                    return (
                      <div
                        className="bbox-hover-preview"
                        style={{
                          left: dispX1,
                          top: dispY1,
                          width: Math.max(10, dispX2 - dispX1),
                          height: Math.max(10, dispY2 - dispY1),
                        }}
                      >
                        <span className="bbox-hover-label">
                          🚗 {hoveredDetectedBox.class_name}（#{hoveredDetectedBox.id + 1}）— クリックして選択
                        </span>
                      </div>
                    );
                  })()}
                </div>

                {/* 커스텀 재생바 및 이벤트 마커 표시 (비디오 화면 바깥 하단으로 이동하여 화면 침범 방지) */}
                <div className="custom-progress-bar">
                  <div className="progress-track" onClick={handleSeek}>
                    {/* 실제 재생 진행률 */}
                    <div
                      className="progress-fill"
                      style={{
                        width: `${selectedVideo.duration ? (currentTime / selectedVideo.duration) * 100 : 0}%`,
                      }}
                    ></div>

                    {/* 이벤트 마커 (타임라인의 빨간 점) */}
                    {selectedVideo.events.map((event) => {
                      const leftPosition = (event.timestamp / selectedVideo.duration) * 100;
                      return (
                        <div
                          key={event.id}
                          className={`event-marker ${currentEventId === event.id ? 'active' : ''}`}
                          style={{ left: `${leftPosition}%` }}
                          title={event.title}
                          onClick={(e) => {
                            e.stopPropagation();
                            seekToEvent(event);
                          }}
                        />
                      );
                    })}
                  </div>
                  <div className="time-labels">

                    <span>0:00</span>
                    <span>{formatTime(selectedVideo.duration)}</span>
                  </div>
                  <div className="player-controls">
                    <button
                      className="control-btn"
                      onClick={togglePlay}
                    >
                      {isPlaying ? "⏸" : "▶"}
                    </button>

                    <label className="control-field icon-control" aria-label="Playback speed">
                      <select value={playbackSpeed} onChange={(e) => setPlaybackSpeed(e.target.value)}>
                        <option value="0.5">0.5x</option>
                        <option value="1">1x</option>
                        <option value="1.5">1.5x</option>
                        <option value="2">2x</option>
                      </select>
                    </label>

                    <label className="control-field volume-field icon-control" aria-label="Volume">
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={volume}
                        onChange={(e) => setVolume(Number(e.target.value))}
                      />
                    </label>

                    <div className="right-controls">
                      <label className="control-field quality-field icon-control" aria-label="Video quality">
                        <select value={quality} onChange={(e) => setQuality(e.target.value)}>
                          <option value="auto">Auto</option>
                          <option value="1080p">1080p</option>
                          <option value="720p">720p</option>
                          <option value="480p">480p</option>
                        </select>
                      </label>
                      <button
                        className={`control-btn theater-btn ${isTheaterMode ? "active" : ""}`}
                        onClick={() => setIsTheaterMode((enabled) => !enabled)}
                        aria-label="シアターモード"
                      >
                        ▭
                      </button>
                      <button
                        className="control-btn fullscreen-btn"
                        onClick={handleToggleFullscreen}
                        aria-label="全画面"
                      >
                        ⛶
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* 바운딩박스 좌표 결과 패널 */}
              {showBBoxPanel && (
                <div className="bbox-result-panel">
                  <div className="bbox-panel-header">
                    <div className="bbox-panel-title">
                      <span className="bbox-panel-icon">📦</span>
                      <span>事故車両のバウンディングボックス座標</span>
                    </div>
                    <div className="bbox-panel-actions">
                      {bboxList.length > 0 && (
                        <button
                          className="bbox-clear-btn"
                          onClick={() => setBboxList([])}
                          title="すべてのボックスを初期化"
                        >
                          初期化
                        </button>
                      )}
                      <button
                        className="bbox-panel-close-btn"
                        onClick={() => setShowBBoxPanel(false)}
                        title="パネルを閉じる"
                      >
                        ✕
                      </button>
                    </div>
                  </div>

                  {bboxList.length === 0 ? (
                    <div className="bbox-empty-hint">
                      <span>🖱️</span>
                      <span>映像上をドラッグしてボックスを描画してください。</span>
                    </div>
                  ) : (
                    <div className="bbox-coord-list">
                      {bboxList.map((box, idx) => (
                        <div key={box.id} className="bbox-coord-row">
                          <span className="bbox-coord-index">#{idx}</span>
                          <code className="bbox-coord-value">
                            (car, {idx}, {box.xmin}, {box.ymin}, {box.xmax}, {box.ymax})
                          </code>
                          <button
                            className="bbox-row-delete-btn"
                            onClick={() => setBboxList((prev) => prev.filter((b) => b.id !== box.id))}
                            title="削除"
                          >
                            ✕
                          </button>
                        </div>
                      ))}
                    </div>
                  )}

                  {bboxList.length > 0 && (
                    <button
                      className="bbox-copy-btn"
                      onClick={() => {
                        const text = bboxList
                          .map((box, idx) => `(car, ${idx}, ${box.xmin}, ${box.ymin}, ${box.xmax}, ${box.ymax})`)
                          .join("\n");
                        navigator.clipboard.writeText(text).then(() => {
                          alert("座標をクリップボードにコピーしました！");
                        });
                      }}
                    >
                      📋 座標をすべてコピー
                    </button>
                  )}
                </div>
              )}
            </section>
          </div>
        </main>
      )}

      {/* 설정 모달 */}
      {isSettingsOpen && (
        <div className="settings-modal-overlay" onClick={() => setIsSettingsOpen(false)}>
          <div className="settings-modal-container" onClick={(e) => e.stopPropagation()}>
            <div className="settings-modal-sidebar">
              <h3>設定</h3>
              <button
                className={`settings-tab-btn ${activeSettingsTab === "account" ? "active" : ""}`}
                onClick={() => setActiveSettingsTab("account")}
              >
                👤 個人情報設定
              </button>
              <button
                className={`settings-tab-btn ${activeSettingsTab === "security" ? "active" : ""}`}
                onClick={() => setActiveSettingsTab("security")}
              >
                🔑 パスワード変更
              </button>
              <button
                className="settings-modal-close-btn"
                onClick={() => setIsSettingsOpen(false)}
              >
                閉じる
              </button>
            </div>

            <div className="settings-modal-content">
              {activeSettingsTab === "account" && (
                <div className="settings-tab-content">
                  <h2>{currentUser?.role === "ADMIN" ? "管理者情報設定" : "個人情報設定"}</h2>
                  <p className="tab-description">
                    {currentUser?.role === "ADMIN"
                      ? "管理者の基本情報を確認・編集できます。"
                      : "会員の基本情報を確認・編集できます。"}
                  </p>

                  <div className="settings-form-group">
                    <label>アカウントID</label>
                    <input type="text" value={currentUsername} disabled className="disabled-input" />
                  </div>

                  <div className="settings-form-group">
                    <label>氏名</label>
                    <input
                      type="text"
                      value={adminRealName}
                      onChange={(e) => setAdminRealName(e.target.value)}
                      placeholder="氏名を入力"
                    />
                  </div>

                  <div className="settings-form-group">
                    <label>連絡先</label>
                    <input
                      type="text"
                      value={adminPhone}
                      onChange={(e) => setAdminPhone(e.target.value)}
                      placeholder="連絡先を入力（例：010-1234-5678）"
                    />
                  </div>

                  <div className="settings-form-group">
                    <label>メールアドレス</label>
                    <input
                      type="email"
                      value={adminEmail}
                      onChange={(e) => setAdminEmail(e.target.value)}
                      placeholder="メールアドレスを入力"
                    />
                  </div>

                  <button
                    className="settings-save-btn"
                    onClick={async () => {
                      if (!adminRealName.trim()) {
                        alert("氏名を入力してください。");
                        return;
                      }
                      try {
                        const updated = await api.updateMe({
                          name: adminRealName.trim(),
                          phone: adminPhone.trim(),
                          email: adminEmail.trim(),
                        });
                        if (onUpdateUser) onUpdateUser(updated);
                        showToast("ユーザー情報を保存しました。", "success");
                        alert("ユーザー情報を保存しました。");
                      } catch (err) {
                        alert(`정보 수정 실패: ${err.message}`);
                      }
                    }}
                  >
                    変更内容を保存
                  </button>
                </div>
              )}

              {activeSettingsTab === "security" && (
                <div className="settings-tab-content">
                  <h2>パスワード変更</h2>
                  <p className="tab-description">システムの安全のため、定期的にパスワードを変更してください。</p>

                  <div className="settings-form-group">
                    <label>現在のパスワード</label>
                    <input
                      type="password"
                      placeholder="現在のパスワードを入力"
                      value={currentPassword}
                      onChange={(e) => setCurrentPassword(e.target.value)}
                    />
                  </div>

                  <div className="settings-form-group">
                    <label>新しいパスワード</label>
                    <input
                      type="password"
                      placeholder="新しいパスワードを入力"
                      value={newPassword}
                      onChange={(e) => setNewPassword(e.target.value)}
                    />
                  </div>

                  <div className="settings-form-group">
                    <label>新しいパスワード確認</label>
                    <input
                      type="password"
                      placeholder="新しいパスワードをもう一度入力"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                    />
                  </div>

                  <button
                    className="settings-save-btn"
                    onClick={async () => {
                      if (!currentPassword || !newPassword || !confirmPassword) {
                        alert("すべての項目を入力してください。");
                        return;
                      }
                      if (newPassword !== confirmPassword) {
                        alert("新しいパスワードが一致しません。");
                        return;
                      }
                      if (newPassword.length < 4) {
                        alert("新しいパスワードは4文字以上で入力してください。");
                        return;
                      }
                      try {
                        await api.changePassword({
                          current_password: currentPassword,
                          new_password: newPassword,
                        });
                        showToast("パスワードを変更しました。", "success");
                        alert("パスワードを変更しました。");
                        setCurrentPassword("");
                        setNewPassword("");
                        setConfirmPassword("");
                        setIsSettingsOpen(false);
                      } catch (err) {
                        alert(`비밀번호 변경 실패: ${err.message}`);
                      }
                    }}
                  >
                    パスワードを変更
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 토스트 알림 */}
      {toast && (
        <div className={`app-toast app-toast-${toast.type}`}>{toast.message}</div>
      )}

      {/* 사고구간 CAM 클립 팝업 */}
      {clipEvent && (
        <div className="clip-modal-overlay" onClick={() => setClipEvent(null)}>
          <div className="clip-modal" onClick={(e) => e.stopPropagation()}>
            <div className="clip-modal-header">
              <span>
                事故区間CAMクリップ · {formatTime(clipEvent.timestamp)}
                {clipEvent.endTimestamp != null
                  ? ` ~ ${formatTime(clipEvent.endTimestamp)}`
                  : ""}
                {clipEvent.prob != null
                  ? ` · ${(clipEvent.prob * 100).toFixed(1)}%`
                  : ""}
              </span>
              <button
                className="clip-modal-close"
                onClick={() => setClipEvent(null)}
              >
                ✕
              </button>
            </div>
            <video
              className="clip-modal-video"
              src={api.clipUrl(clipEvent.id)}
              controls
              autoPlay
            />
          </div>
        </div>
      )}

      {/* 영상 업로드 모달 */}
      {showUpload && (
        <UploadModal
          onClose={() => setShowUpload(false)}
          onUploaded={async () => {
            setShowUpload(false);
            showToast("업로드 완료", "success");
            await loadVideos();
          }}
          onError={(msg) => showToast(msg, "error")}
        />
      )}
      {/* 동일 날짜 여러 영상 선택 모달 */}
      {datePickerVideos && (
        <DateVideosModal
          dateText={datePickerVideos.dateText}
          videos={datePickerVideos.videos}
          onClose={() => setDatePickerVideos(null)}
          onSelectVideo={(video) => {
            setDatePickerVideos(null);
            handleWatchVideo(video);
          }}
        />
      )}
    </div>
  );
}

function UploadModal({ onClose, onUploaded, onError }) {
  const [filesList, setFilesList] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({ current: 0, total: 0, currentName: "" });
  const [dragActive, setDragActive] = useState(false);

  const fileInputRef = useRef(null);

  const addFiles = (newFiles) => {
    if (!newFiles || newFiles.length === 0) return;
    const todayStr = formatDate(new Date());
    const items = Array.from(newFiles).map((file, idx) => ({
      id: `${Date.now()}_${idx}_${Math.random().toString(36).substring(2, 7)}`,
      file,
      recordingDate: todayStr,
    }));
    setFilesList((prev) => [...prev, ...items]);
  };

  const handleFileChange = (e) => {
    addFiles(e.target.files);
    if (e.target) e.target.value = "";
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addFiles(e.dataTransfer.files);
    }
  };

  const removeFile = (id) => {
    setFilesList((prev) => prev.filter((item) => item.id !== id));
  };

  const updateFileDate = (id, dateStr) => {
    setFilesList((prev) =>
      prev.map((item) => (item.id === id ? { ...item, recordingDate: dateStr } : item))
    );
  };

  const updateAllDates = (dateStr) => {
    setFilesList((prev) => prev.map((item) => ({ ...item, recordingDate: dateStr })));
  };

  const handleUpload = async () => {
    if (filesList.length === 0) {
      onError("アップロードする映像ファイルを選択してください。");
      return;
    }
    setUploading(true);
    let successCount = 0;
    let failCount = 0;

    for (let i = 0; i < filesList.length; i += 1) {
      const item = filesList[i];
      setUploadProgress({
        current: i + 1,
        total: filesList.length,
        currentName: item.file.name,
      });
      try {
        await api.uploadVideo(item.file, item.recordingDate || null);
        successCount += 1;
      } catch (e) {
        failCount += 1;
        console.error(`Failed to upload ${item.file.name}:`, e);
      }
    }

    setUploading(false);
    if (successCount > 0) {
      if (failCount > 0) {
        onError(`${successCount}本のアップロード完了（${failCount}本失敗）`);
      }
      onUploaded();
    } else {
      onError("すべての映像ファイルのアップロードに失敗しました。");
    }
  };

  return (
    <div className="settings-modal-overlay" onClick={onClose}>
      <div className="upload-modal multi-upload-modal" onClick={(e) => e.stopPropagation()}>
        <div className="upload-modal-header">
          <div>
            <h2>映像を一括アップロード</h2>
            <p className="tab-description">分析するCCTV録画映像をアップロードします。複数ファイルを選択し、録画日を個別に指定できます。</p>
          </div>
          <button className="clip-modal-close" onClick={onClose} disabled={uploading}>✕</button>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          multiple
          style={{ display: "none" }}
          onChange={handleFileChange}
        />

        {filesList.length === 0 ? (
          <div
            className={`upload-dropzone ${dragActive ? "drag-active" : ""}`}
            onDragEnter={handleDrag}
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="dropzone-icon">📹</div>
            <p className="dropzone-title">クリックして映像ファイルを選択（複数選択可）</p>
            <p className="dropzone-sub">またはここに映像ファイルをドラッグ＆ドロップ</p>
          </div>
        ) : (
          <div className="upload-file-list-container">
            <div className="upload-file-list-toolbar">
              <span className="file-count-badge">選択した映像：<strong>{filesList.length}本</strong></span>
              <div className="file-toolbar-actions">
                <button
                  type="button"
                  className="add-more-files-btn"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={uploading}
                >
                  + ファイルを追加
                </button>
                <button
                  type="button"
                  className="clear-all-files-btn"
                  onClick={() => setFilesList([])}
                  disabled={uploading}
                >
                  すべて削除
                </button>
              </div>
            </div>

            <div className="bulk-date-bar">
              <span>録画日を一括指定：</span>
              <input
                type="date"
                max={formatDate(new Date())}
                onChange={(e) => {
                  if (e.target.value) updateAllDates(e.target.value);
                }}
                disabled={uploading}
              />
              <span className="bulk-date-tip">（ファイルごとに個別指定できます）</span>
            </div>

            <div className="upload-file-scroll-list">
              {filesList.map((item) => (
                <div key={item.id} className="upload-file-item">
                  <div className="file-item-info">
                    <span className="file-item-icon">🎥</span>
                    <div className="file-item-name-group">
                      <span className="file-item-name" title={item.file.name}>{item.file.name}</span>
                      <span className="file-item-size">{(item.file.size / (1024 * 1024)).toFixed(1)} MB</span>
                    </div>
                  </div>

                  <div className="file-item-date-group">
                    <label>録画日：</label>
                    <input
                      type="date"
                      value={item.recordingDate}
                      max={formatDate(new Date())}
                      onChange={(e) => updateFileDate(item.id, e.target.value)}
                      disabled={uploading}
                    />
                  </div>

                  <button
                    type="button"
                    className="file-item-remove-btn"
                    onClick={() => removeFile(item.id)}
                    disabled={uploading}
                    title="파일 삭제"
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {uploading && (
          <div className="upload-progress-box">
            <div className="upload-progress-info">
              <span>アップロード中（{uploadProgress.current}/{uploadProgress.total}）</span>
              <span className="upload-progress-filename">{uploadProgress.currentName}</span>
            </div>
            <div className="analysis-progress-track">
              <div
                className="analysis-progress-fill"
                style={{ width: `${(uploadProgress.current / uploadProgress.total) * 100}%` }}
              />
            </div>
          </div>
        )}

        <div className="upload-modal-actions">
          <button
            className="upload-submit-btn"
            onClick={handleUpload}
            disabled={uploading || filesList.length === 0}
          >
            {uploading
              ? `アップロード中（${uploadProgress.current}/${uploadProgress.total}）…`
              : `${filesList.length > 0 ? `${filesList.length}本 ` : ""}映像をアップロード`}
          </button>
          <button className="upload-cancel-btn" onClick={onClose} disabled={uploading}>
            キャンセル
          </button>
        </div>
      </div>
    </div>
  );
}

function DateVideosModal({ dateText, videos, onClose, onSelectVideo }) {
  if (!videos || videos.length === 0) return null;

  return (
    <div className="settings-modal-overlay" onClick={onClose}>
      <div className="date-videos-modal" onClick={(e) => e.stopPropagation()}>
        <div className="date-videos-header">
          <div>
            <h2>📅 {dateText} 録画映像一覧（{videos.length}本）</h2>
            <p className="tab-description">
              この日には{videos.length}本の録画映像があります。視聴する映像を選択してください。
            </p>
          </div>
          <button className="clip-modal-close" onClick={onClose}>✕</button>
        </div>

        <div className="date-videos-scroll-list">
          {videos.map((v) => (
            <div key={v.id} className="date-video-card-item" onClick={() => onSelectVideo(v)}>
              <div className="date-video-thumb-wrapper">
                <img
                  className="video-thumbnail-img"
                  src={api.thumbnailUrl(v.id)}
                  alt={`${v.date} 썸네일`}
                  onError={(e) => { e.currentTarget.style.display = "none"; }}
                />
                <span className="event-count-badge">イベント {v.events.length}件</span>
              </div>
              <div className="date-video-card-info">
                <h3>{getVideoDisplayTitle(v, videos)}</h3>
                <div className="date-video-card-meta">
                  <span>⏱ 映像の長さ：{formatTime(v.duration)}</span>
                  <span>📍 場所：{v.camera}</span>
                  <span>⏱ 開始：{v.startTime}</span>
                </div>
              </div>
              <button className="date-video-play-btn">
                ▶ 視聴する
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function RequireAuth({ children }) {
  return getToken() ? children : <Navigate to="/login" replace />;
}

export default function App() {
  const [currentUser, setCurrentUser] = useState(() => getStoredUser());
  const [user, setUser] = useState(() => !!getToken());
  const [isInitializing, setIsInitializing] = useState(true);

  useEffect(() => {
    if (getToken()) {
      api.getMe().then((u) => {
        setCurrentUser(u);
        saveAuth(getToken(), u);
      }).catch(() => { });
    }
    const timer = setTimeout(() => {
      setIsInitializing(false);
    }, 1000);
    return () => clearTimeout(timer);
  }, []);

  const handleUpdateUser = (updatedUser) => {
    setCurrentUser(updatedUser);
    saveAuth(getToken(), updatedUser);
  };

  const handleLogout = () => {
    clearAuth();
    setCurrentUser(null);
    setUser(false);
  };

  if (isInitializing) {
    return <AppLoadingScreen />;
  }

  return (

    <BrowserRouter>
      <Routes>
        <Route
          path="/login"
          element={
            <LoginPage
              onLogin={(loggedUser) => {
                setCurrentUser(loggedUser);
                setUser(true);
              }}
            />
          }
        />
        <Route path="/signup" element={<SignupPage />} />
        <Route
          path="/videos"
          element={
            <RequireAuth>
              <Dashboard
                currentUser={currentUser}
                onUpdateUser={handleUpdateUser}
                onLogout={handleLogout}
                view="list"
              />
            </RequireAuth>
          }
        />
        <Route
          path="/videos/:videoId"
          element={
            <RequireAuth>
              <Dashboard
                currentUser={currentUser}
                onUpdateUser={handleUpdateUser}
                onLogout={handleLogout}
                view="watch"
              />
            </RequireAuth>
          }
        />
        <Route
          path="/analytics"
          element={
            <RequireAuth>
              <Dashboard
                currentUser={currentUser}
                onUpdateUser={handleUpdateUser}
                onLogout={handleLogout}
                view="analytics"
              />
            </RequireAuth>
          }
        />
        <Route path="*" element={<Navigate to={user ? "/videos" : "/login"} replace />} />
      </Routes>
    </BrowserRouter>
  );
}
