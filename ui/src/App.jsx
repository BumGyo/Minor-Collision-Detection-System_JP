import { useCallback, useMemo, useRef, useState, useEffect, useLayoutEffect, useSyncExternalStore } from "react";
import replay5Icon from "./assets/icons/replay-5.svg";
import forward5Icon from "./assets/icons/forward-5.svg";
import playArrowIcon from "./assets/icons/play_arrow.svg";
import pauseIcon from "./assets/icons/pause.svg";
import volumeUpIcon from "./assets/icons/volume_up.svg";
import volumeOffIcon from "./assets/icons/volume_off.svg";
import volumeDownIcon from "./assets/icons/volume_down.svg";
import fastForwardIcon from "./assets/icons/fast_forward.svg";
import fastRewindIcon from "./assets/icons/fast_rewind.svg";
import speedIcon from "./assets/icons/speed.svg";
import theatersIcon from "./assets/icons/theaters.svg";
import fullscreenIcon from "./assets/icons/fullscreen.svg";
import fullscreenExitIcon from "./assets/icons/fullscreen_exit.svg";
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
import "./incident-ui.css";
import "./premium-ui.css";
import "./dashboard-dark.css"; // ダークモード統一 — 必ず premium-ui.css の後に配置
import "./siot-refresh.css";

const VIDEO_FEEDBACK_ICONS = {
  play: playArrowIcon,
  pause: pauseIcon,
  replay5: replay5Icon,
  forward5: forward5Icon,
  fastRewind: fastRewindIcon,
  fastForward: fastForwardIcon,
  volumeUp: volumeUpIcon,
  volumeDown: volumeDownIcon,
  volumeOff: volumeOffIcon,
  speed: speedIcon,
  theater: theatersIcon,
  fullscreen: fullscreenIcon,
  fullscreenExit: fullscreenExitIcon,
};

function InterfaceIcon({ name }) {
  const paths = {
    user: <><circle cx="12" cy="8" r="3" /><path d="M5 21v-2a7 7 0 0 1 14 0v2" /></>,
    lock: <><rect x="5" y="10" width="14" height="11" rx="3" /><path d="M8 10V7a4 4 0 0 1 8 0v3M12 14v3" /></>,
    chart: <><path d="M4 4v16h17M8 15v-4M13 15V6M18 15V9" /></>,
    moon: <path d="M20 14A8 8 0 0 1 10 4a8 8 0 1 0 10 10Z" />,
    exit: <><path d="M10 4H5v16h5M10 12h11M17 8l4 4-4 4" /></>,
  };
  return <svg className="interface-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

// 動画カードの ··· メニュー — 「動画情報確認」は画面側の情報ポップアップ(onShowInfo、アップロード日時等を含む)を開き、
// 「動画削除」はここで確認ダイアログを表示して削除する
function VideoCardActions({ video, title, onShowInfo, onDeleted, onError }) {
  const [open, setOpen] = useState(false);
  const [dialog, setDialog] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const containerRef = useRef(null);
  const triggerRef = useRef(null);
  useEffect(() => {
    if (!open && !dialog) return;
    const outside = (event) => {
      if (!containerRef.current?.contains(event.target)) setOpen(false);
    };
    const escape = (event) => {
      if (event.key === "Escape" && !busy) {
        setOpen(false); setDialog(null); triggerRef.current?.focus();
      }
    };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape);
    return () => { document.removeEventListener("pointerdown", outside); document.removeEventListener("keydown", escape); };
  }, [open, dialog, busy]);
  const close = () => { if (!busy) { setDialog(null); triggerRef.current?.focus(); } };
  return <div className="video-card-actions" ref={containerRef} onClick={(e) => e.stopPropagation()}>
    {/* 三点リーダーは文字(···)ではなくSVG — フォントやOSによってドットの位置がずれるのを防止 */}
    <button className="video-more-trigger" ref={triggerRef} aria-label={`${title} メニュー`} aria-expanded={open} onClick={() => setOpen(!open)}>
      <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><circle cx="5" cy="12" r="2" /><circle cx="12" cy="12" r="2" /><circle cx="19" cy="12" r="2" /></svg>
    </button>
    {open && <div className="video-card-menu">
      <button onClick={() => { setOpen(false); onShowInfo(); }}><InterfaceIcon name="chart" /> 動画情報を確認</button>
      <button className="video-menu-delete" onClick={() => { setOpen(false); setError(""); setDialog("delete"); }}><TrashIcon size={17} /> 動画を削除</button>
    </div>}
    {dialog && <div className="settings-modal-overlay" onClick={close}>
      <div className="confirm-modal video-card-dialog" role="alertdialog" aria-modal="true" aria-label="動画削除の確認" onClick={(e) => e.stopPropagation()}>
        <span className="incident-eyebrow" translate="no">DELETE VIDEO</span>
        <h2>この動画を削除しますか？</h2>
        {/* 削除対象の動画をカード表示 — サムネイル・タイトル・録画日と再生時間で対象動画を一目で識別可能に */}
        <div className="video-dialog-target">
          <img className="video-dialog-thumb" src={api.thumbnailUrl(video.id)} alt="" onError={(e) => { e.currentTarget.style.visibility = "hidden"; }} />
          <div className="video-dialog-target-text">
            <strong className="video-dialog-title" title={title}>{title}</strong>
            <span className="video-dialog-meta">{[video.date, formatTime(video.duration)].filter(Boolean).join(" · ")}</span>
          </div>
        </div>
        <p className="confirm-modal-desc">原本動画、分析履歴、事故クリップがすべて削除されます。削除後は復元できません。</p>
        {error && <p className="video-delete-error" role="alert">{error}</p>}
        <div className="video-dialog-actions"><button autoFocus disabled={busy} onClick={close}>キャンセル</button>
          {dialog === "delete" && <button className="video-delete-confirm" disabled={busy} onClick={async () => {
            setBusy(true); setError("");
            try { await api.deleteVideo(video.id); onDeleted(video.id); }
            catch (e) { setError(e.message); onError?.(e.message); setBusy(false); }
          }}>{busy ? "削除中…" : "動画を削除"}</button>}
        </div>
      </div>
    </div>}
  </div>;
}

function TeamFooter({ animated = false }) {
  return <footer className={`siot-team-footer ${animated ? "scroll-reveal" : ""}`}>
    <div className="team-footer-brand"><strong translate="no">SIOT<span>.</span></strong><p>忠北大学校 ソフトウェア学部</p><small translate="no">Parking incident intelligence</small></div>
    <div className="team-footer-people"><span className="team-footer-label" translate="no">THE TEAM</span><p><span>チームリーダー</span> イ・ジョンジュ</p><p><span>チームメンバー</span> パク・エマニュエル・ジョン, ペ・ギウォン, ソ・ボムギョ</p></div>
    <a className="team-footer-github" href="https://github.com/CBNU-capstone-26/Minor-Collision-Detection-System" target="_blank" rel="noreferrer">GitHub <span aria-hidden="true">↗</span><small>プロジェクトリポジトリ</small></a>
  </footer>;
}

// 線画のゴミ箱アイコン — 文字色(currentColor)に追従し、通常グレー → ホバー時レッド。
// (🗑 絵文字は固有色のためhover色が効かず、OSごとに形状が異なる)
function TrashIcon({ size = 14 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M3 6h18" />
      <path d="M8 6V4h8v2" />
      <path d="M19 6l-1 14H6L5 6" />
      <path d="M10 11v6" />
      <path d="M14 11v6" />
    </svg>
  );
}

// 再生コントロールアイコン — 絵文字(▶ ⏸ ⛶)はOSやフォントによって形状やサイズが異なるためSVGで描画
const PLAYER_ICON_PATHS = {
  play: <path d="M8 5.5v13l10.5-6.5z" fill="currentColor" />,
  pause: (
    <>
      <rect x="6.5" y="5" width="3.5" height="14" rx="1" fill="currentColor" stroke="none" />
      <rect x="14" y="5" width="3.5" height="14" rx="1" fill="currentColor" stroke="none" />
    </>
  ),
  volumeHigh: (
    <>
      <path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z" fill="currentColor" />
      <path d="M15.5 9a4 4 0 0 1 0 6" />
      <path d="M18 6.5a7.5 7.5 0 0 1 0 11" />
    </>
  ),
  volumeLow: (
    <>
      <path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z" fill="currentColor" />
      <path d="M15.5 9a4 4 0 0 1 0 6" />
    </>
  ),
  volumeMute: (
    <>
      <path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z" fill="currentColor" />
      <path d="M16 9.5l5 5M21 9.5l-5 5" />
    </>
  ),
  theater: <rect x="2.5" y="6.5" width="19" height="11" rx="1.5" />,
  settings: <><path d="M4 6h10m4 0h2M4 12h4m4 0h8M4 18h10m4 0h2" /><circle cx="16" cy="6" r="2" /><circle cx="10" cy="12" r="2" /><circle cx="16" cy="18" r="2" /></>,
  fullscreen: <path d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5" />,
  exitFullscreen: <path d="M9 4v5H4M20 9h-5V4M15 20v-5h5M4 15h5v5" />,
};

function PlayerIcon({ name, size = 18 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {PLAYER_ICON_PATHS[name]}
    </svg>
  );
}

// グリッドの現在のカラム数 — CSS(auto-fill)が画面幅・倍率に応じて定めた値を読み取る。el変更時やリサイズ時に再取得
function useGridColumns(el, fallback = 3) {
  const [cols, setCols] = useState(fallback);
  useEffect(() => {
    if (!el || typeof ResizeObserver === "undefined") return undefined;
    const read = () => {
      const n = getComputedStyle(el).gridTemplateColumns.split(" ").filter(Boolean).length;
      if (n > 0) setCols(n);
    };
    read();
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, [el]);
  return cols;
}

// 現在の再生時刻に表示する事故クリップを検索。重複する分析がある場合は選択されたクリップ、なければ最新分析を採用。
function clipEventAt(events, t, selectedId) {
  const inSeg = events.filter((e) => e.hasClip && e.clipStartSec != null
    && t >= e.startSec && t <= Math.max(e.startSec, e.endSec ?? e.startSec));
  return inSeg.find((e) => e.id === selectedId)
    || inSeg.reduce((best, e) => (!best || (e.analysisNo ?? 0) > (best.analysisNo ?? 0)
      || ((e.analysisNo ?? 0) === (best.analysisNo ?? 0) && e.id > best.id) ? e : best), null);
}

const EVENT_PREROLL_SEC = 1;

function clipEventsAhead(events, t, selectedId, lead) {
  const times = [t];
  events.forEach((e) => {
    if (!e.hasClip || e.clipStartSec == null) return;
    [e.startSec, Math.max(e.startSec, e.endSec ?? e.startSec) + 0.001].forEach((b) => {
      if (b > t && b <= t + lead) times.push(b);
    });
  });
  const out = [];
  times.forEach((x) => {
    const ev = clipEventAt(events, x, selectedId);
    if (ev && !out.includes(ev)) out.push(ev);
  });
  return out;
}

let toastSeq = 0; // トースト通知ID (showToast)
let bboxSeq = 0; // 選択車両ボックスID

// 画面幅条件(CSSメディアクエリ)の現在の一致状態 — ウィンドウリサイズ時に再描画
function useMediaQuery(query) {
  const subscribe = useCallback(
    (onChange) => {
      const mql = window.matchMedia(query);
      mql.addEventListener("change", onChange);
      return () => mql.removeEventListener("change", onChange);
    },
    [query]
  );
  return useSyncExternalStore(subscribe, () => window.matchMedia(query).matches);
}

// 検索アイコン — 塗りのない線画の虫眼鏡 (絵文字🔍はOSによって色や形状が異なるため)
function SearchIcon({ size = 16 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="11" cy="11" r="7" />
      <path d="M20 20l-4-4" />
    </svg>
  );
}

// ログインの有効期限が切れて戻った場合(/login?expired=1)、ログイン画面を表示して案内する
const AUTH_EXPIRED_MSG = "ログインの有効期限が切れました。再度ログインしてください。";
const cameFromExpiredLogin = () => new URLSearchParams(window.location.search).has("expired");

// ログイン・会員登録ポップアップの非表示アニメーション時間 — App.css の public-overlay-out・public-modal-out と同期
const MODAL_EXIT_MS = 200;
const prefersReducedMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

function animateFullscreenRect(element, fromRect, toRect) {
  if (!element || !fromRect || !toRect || prefersReducedMotion()) return;
  if (!fromRect.width || !fromRect.height || !toRect.width || !toRect.height) return;
  const x = fromRect.left - toRect.left;
  const y = fromRect.top - toRect.top;
  const scaleX = fromRect.width / toRect.width;
  const scaleY = fromRect.height / toRect.height;
  element.style.transformOrigin = "top left";
  element.style.transition = "none";
  element.style.transform = `translate3d(${x}px, ${y}px, 0) scale(${scaleX}, ${scaleY})`;
  element.getBoundingClientRect();
  requestAnimationFrame(() => {
    element.style.transition = "transform 360ms cubic-bezier(0.22, 1, 0.36, 1)";
    element.style.transform = "translate3d(0, 0, 0) scale(1, 1)";
  });
  window.setTimeout(() => {
    element.style.transition = "none";
    element.style.transform = "";
    element.style.transformOrigin = "";
    element.getBoundingClientRect();
    element.style.transition = "";
  }, 400);
}

function LoginPage({ onLogin }) {
  const isTouchDevice = useMediaQuery("(pointer: coarse)");
  const [id, setId] = useState("");
  const [pw, setPw] = useState("");
  const [error, setError] = useState(() => (cameFromExpiredLogin() ? AUTH_EXPIRED_MSG : ""));
  const [isLoginOpen, setIsLoginOpen] = useState(cameFromExpiredLogin);
  const [isSignupOpen, setIsSignupOpen] = useState(false);
  const [isModalClosing, setIsModalClosing] = useState(false);
  const modalCloseTimerRef = useRef(null);
  const navigate = useNavigate();

  // ポップアップはCSSアニメーションで開き、閉じる際は非表示アニメーション完了後に実際にアンマウントする
  // (「動きを減らす」設定時はアニメーションなしで即時閉じる)
  const openLogin = () => {
    clearTimeout(modalCloseTimerRef.current);
    setIsModalClosing(false);
    setIsSignupOpen(false);
    setIsLoginOpen(true);
  };
  const goToLoginHome = () => {
    navigate("/login", { replace: true });
    window.scrollTo({ top: 0, behavior: prefersReducedMotion() ? "auto" : "smooth" });
  };
  const closeModal = () => {
    const finish = () => {
      setIsLoginOpen(false);
      setIsSignupOpen(false);
      setIsModalClosing(false);
    };
    clearTimeout(modalCloseTimerRef.current);
    if (prefersReducedMotion()) {
      finish();
      return;
    }
    setIsModalClosing(true);
    modalCloseTimerRef.current = setTimeout(finish, MODAL_EXIT_MS);
  };

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            // Optional: unobserve if you only want it to animate once
            // observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.1, rootMargin: "0px 0px -50px 0px" }
    );

    const elements = document.querySelectorAll(".scroll-reveal");
    elements.forEach((el) => observer.observe(el));

    return () => observer.disconnect();
  }, []);

  const handleLogin = async () => {
    setError("");
    try {
      const res = await api.login({ username: id, password: pw });
      document.activeElement?.blur?.();
      saveAuth(res.token, res.user);
      onLogin(res.user);
      navigate("/videos");
      window.requestAnimationFrame(() => window.scrollTo({ top: 0, left: 0, behavior: "instant" }));
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <div className="public-login-page">
      <header className="public-navbar">
        <button className="public-brand" onClick={goToLoginHome} aria-label="SIOT ホームへ移動">
          <span className="workspace-wordmark" translate="no">SI<span className="workspace-lens">O</span>T<span className="workspace-brand-dot">.</span></span>
        </button>
        <div className="public-nav-meta">
          <nav className="premium-nav-links" aria-label="サービス紹介"><a href="#overview" translate="no">SIOT</a><a href="#review-preview">クリップ確認</a><a href="#why-siot">主要機能</a></nav>
          <button className="public-login-trigger" translate="no" onClick={openLogin}>
            LOGIN <span aria-hidden="true">↗</span>
          </button>
        </div>
      </header>

      <main className="public-main">
        <section className="public-hero" id="overview">
          <div className="public-hero-copy scroll-reveal">
            <p className="public-kicker" translate="no"><span className="premium-status-dot" /> SIOT · PARKING INTELLIGENCE</p>
            <h1>見落とされがちな瞬間。<br /><em>鮮明になる手掛かり。</em></h1>
            <p className="public-hero-description">
              膨大なCCTV映像から、確認が必要な瞬間へ。
              SIOTが駐車中の接触疑いシーンを検出し、的確な確認作業を支援します。
            </p>
            <div className="public-hero-actions">
              <button className="public-primary-btn" onClick={openLogin}>
                分析を開始する <span aria-hidden="true">→</span>
              </button>
              <a className="public-text-link" href="#why-siot">SIOTの特長を見る <span aria-hidden="true">↓</span></a>
            </div>
          </div>

          <div className="public-hero-panel scroll-reveal" aria-label="SIOTの分析フロー">
            <div className="public-panel-topline" translate="no">
              <span>INCIDENT PREVIEW</span>
              <span className="public-live-dot">● DEMO</span>
            </div>
            <div className="public-scan-frame">
              <div className="public-scan-grid" />
              <div className="public-scan-road">
                <span className="public-car car-one" />
                <span className="public-car car-two" />
                <span className="public-car car-three" />
                <div className="public-detection-box box-blue-car">
                  <span className="public-detected-label" translate="no">DETECTED</span>
                </div>
              </div>
              <div className="public-scan-caption">SIMULATION / 実際の分析結果ではなくサンプルです</div>
            </div>


            <div className="premium-preview-timeline" aria-hidden="true"><span /><span /><span /><i /></div>
            <div className="public-panel-footer">
              <span>疑い区間の検知</span>
              <strong>→ 日付別イベントとして整理</strong>
            </div>
          </div>
        </section>

        <section className="public-proof-strip scroll-reveal" aria-label="SIOTのコア機能">
          <div><strong>01</strong><span>動画アップロード</span></div>
          <div><strong>02</strong><span>疑いシーンの分析</span></div>
          <div><strong>03</strong><span>事象別の確認</span></div>
          <p>AIが映像をスクリーニングし、管理者は必要なシーンに集中できます。</p>
        </section>

        <section className="premium-showcase scroll-reveal" id="review-preview">
          <div className="premium-showcase-copy"><p className="public-kicker" translate="no">SIOT. A CLOSER LOOK.</p><h2>見落としやすい接触。<br /><span>振り返ることができる瞬間。</span></h2><p>長時間の映像から疑い区間へ。<br />クリップと原本を同時に照合する明瞭な確認体験。</p><a className="public-text-link" href="#why-siot">確認フローを見る <span aria-hidden="true">↗</span></a></div>
          <div className="premium-review-window" aria-label="事故疑い区間の確認画面サンプル">
            <div className="premium-window-top"><span><i /><i /><i /></span><b translate="no">SIOT / CLIP REVIEW</b><small>サンプル画面</small></div>
            <div className="premium-window-scene"><div className="premium-parking-lines" /><div className="premium-scene-car parked" /><div className="premium-scene-car passing" /><div className="premium-scene-target"><span>車両選択</span></div><span className="premium-scene-note">接触疑い区間 · 原本確認が必要</span></div>
            <div className="premium-window-bottom"><span aria-hidden="true">▶</span><div><i /></div><span>00:12 / 00:48</span></div>
          </div>
        </section>
        <section className="premium-story" id="why-siot">
          <div className="premium-story-heading scroll-reveal"><p className="public-kicker" translate="no">LESS SEARCHING. MORE CLARITY.</p><h2>長時間の記録は短縮。<br /><span>重要な瞬間は詳細に。</span></h2><p>車両を指定し、疑い区間を検出し、原本で確認する。<br />煩雑だった映像確認を一つのスムーズなフローに統合します。</p></div>
          <div className="premium-feature-grid">
            <article className="premium-feature scroll-reveal"><span className="premium-feature-index" translate="no">01 / FOCUS</span><div className="premium-focus-art" aria-hidden="true"><span /><i /><b /></div><h3>1台にフォーカスする視線。</h3><p>自動車両検出と手動指定により、<br />確認対象の車両を的確に選択できます。</p><span className="premium-feature-tag">車両検出 · エリア指定</span></article>
            <article className="premium-feature scroll-reveal"><span className="premium-feature-index" translate="no">02 / DISCOVER</span><div className="premium-wave-art" aria-hidden="true">{Array.from({length: 23}, (_, i) => <i key={i} style={{"--bar-height": `${20 + ((i * 37) % 65)}%`, "--bar-delay": `${i * 65}ms`}} />)}</div><h3>記録の中に潜む、疑わしい瞬間。</h3><p>選択した車両の接触疑いシーンを<br />分析し、クリップ単位で確認できます。</p><span className="premium-feature-tag">疑い区間の分析 · クリップ確認</span></article>
            <article className="premium-feature scroll-reveal"><span className="premium-feature-index" translate="no">03 / REVIEW</span><div className="premium-review-art" aria-hidden="true"><span>00:12 <b>疑い区間</b></span><span>00:28 <b>原本確認</b></span><span>00:41 <b>クリップ確認</b></span></div><h3>正確な判断のための、より確かな文脈。</h3><p>分析履歴と原本クリップを併せて確認し、<br />前後の状況・文脈を検証できます。</p><span className="premium-feature-tag">分析履歴 · 原本ダウンロード</span></article>
          </div>
        </section>
        <section className="premium-closing scroll-reveal"><span className="public-kicker" translate="no">YOUR NEXT CLEAR MOMENT</span><h2>これからは、必要なシーンだけに<br />集中できます。</h2><button className="public-primary-btn" onClick={openLogin}>SIOTを始める <span aria-hidden="true">↗</span></button><p>AI検知は確認作業を支援する補助情報です。最終確認は原本動画にて行ってください。</p></section>
      </main>

      <TeamFooter animated />

      {/* ログイン・会員登録が共通のオーバーレイを使用 — 遷移時のチラつきを防止 */}
      {(isLoginOpen || isSignupOpen) && (
        <div className={`login-modal-overlay ${isModalClosing ? "closing" : ""}`} onClick={closeModal}>
          {isLoginOpen ? (
            <div className="login-modal" role="dialog" aria-modal="true" aria-labelledby="login-modal-title" onClick={(e) => e.stopPropagation()}>
              <button className="login-modal-close" onClick={closeModal} aria-label="ログイン画面を閉じる">×</button>
              <p className="public-kicker" translate="no">SECURE ACCESS</p>
              <h2 id="login-modal-title">ダッシュボード ログイン</h2>
              <p className="login-modal-description">分析済み動画および事故疑いイベントをご確認ください。</p>

              <div className="input-group">
                <input
                  type="text"
                  placeholder="ユーザー名"
                  value={id}
                  onChange={(e) => setId(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleLogin()}
                  autoFocus={!isTouchDevice}
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
                  <input type="checkbox" /> Remember me
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
                会員登録
              </button>
            </div>
          ) : (
            <SignupModal onClose={closeModal} onBackToLogin={() => {
              setIsSignupOpen(false);
              setIsLoginOpen(true);
            }} />
          )}
        </div>
      )}
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
      setError("ユーザー名、氏名、パスワードは必須です。");
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

  // 背景(login-modal-overlay)はLoginPageがログイン画面と共用
  return (
    <div className="login-modal signup-modal" role="dialog" aria-modal="true" aria-labelledby="signup-modal-title" onClick={(e) => e.stopPropagation()}>
      <button className="login-modal-close" onClick={onClose} aria-label="会員登録画面を閉じる">×</button>
      <p className="public-kicker" translate="no">CREATE ACCOUNT</p>
      <h2 id="signup-modal-title">会員登録</h2>
      <p className="login-modal-description">SIOT動画分析ダッシュボードのアカウントを作成します。</p>

      {done ? (
        <div className="signup-complete-state">
          <strong>登録が完了しました。</strong>
          <span>ログインしてダッシュボードをご利用いただけます。</span>
          <button className="login-submit-btn" onClick={onBackToLogin}>ログイン画面へ</button>
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
              <label className="signup-field-label" htmlFor="signup-username">ユーザー名 <span>*</span></label>
              <div className="input-group">
                <input id="signup-username" type="text" placeholder="ログインに使用するユーザー名" value={form.username} onChange={update("username")} autoCapitalize="off" autoCorrect="off" />
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
      setError("ユーザー名、氏名、パスワードは必須です。");
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
          <h1>会員登録</h1>
          <p>管理者アカウントを作成してシステムに接続してください。</p>
        </div>
      </div>

      <div className="login-right-panel">
        <div className="login-form-wrapper">
          <h2>会員登録</h2>
          <p className="login-subtitle">新規アカウントを作成します。</p>

          <div className="input-group">
            <input type="text" placeholder="氏名 *" value={form.name} onChange={update("name")}
              lang="ja" style={{ imeMode: "active" }} />
          </div>
          <div className="input-group">
            <input type="text" placeholder="ユーザー名 *" value={form.username} onChange={update("username")}
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
          {done && <p className="login-success-text">登録完了！ログイン画面へ移動します。</p>}

          <button className="login-submit-btn" onClick={handleSignup}>登録する</button>
          <button className="login-signup-btn" onClick={() => navigate("/login")}>
            ログインに戻る
          </button>
        </div>
      </div>
    </div>
  );
}

// ユーティリティ関数群
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

// 分析が完了して事故疑いクリップが生成された日時 — ['YYYY-MM-DD', '04時09分'] (閲覧者のローカルタイムゾーン)
function formatAnalyzedAt(iso) {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  const p = (n) => String(n).padStart(2, "0");
  return [`${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`, `${p(d.getHours())}時${p(d.getMinutes())}分`];
}

// 所要時間 — '45秒', '3分12秒', '1時間5分'
function formatDuration(sec) {
  if (sec == null || !Number.isFinite(sec) || sec < 0) return "";
  const s = Math.round(sec);
  if (s < 60) return `${s}秒`;
  const m = Math.floor(s / 60);
  if (m < 60) return s % 60 ? `${m}分${s % 60}秒` : `${m}分`;
  return `${Math.floor(m / 60)}時間${m % 60}分`;
}

const HISTORY_STATUS = {
  SUCCESS: { label: "完了", cls: "success" },
  FAILURE: { label: "失敗", cls: "failure" },
  CANCELLED: { label: "キャンセル", cls: "cancelled" },
  PENDING: { label: "待機中", cls: "active" },
  PROCESSING: { label: "分析中", cls: "active" },
};

// 動画再生画面右側: この動画の分析履歴 (最新の分析が上)
// history: サーバーが返した分析履歴, events: 現在残っているクリップ(分析番号別に集計),
// progressText(taskId): 進行中タスクの進捗テキスト(なければnull)
function AnalysisHistoryPanel({ history, events, progressText, onSelect, activeNo, onClearHistory, clearingHistory }) {
  const remain = {};
  events.forEach((e) => {
    if (e.analysisNo != null) remain[e.analysisNo] = (remain[e.analysisNo] || 0) + 1;
  });

  const resultOf = (h) => {
    const left = remain[h.no] || 0;
    if (h.status === "SUCCESS") {
      if (h.eventCount == null) {
        // 履歴機能実装前の分析 — 検出数が不明なため残存数のみ表示
        return left > 0 ? { text: `事故疑いクリップ ${left}件` } : { text: "残存クリップなし", cls: "muted" };
      }
      if (h.eventCount === 0) return { text: "事故疑い区間なし", cls: "muted" };
      const deleted = h.eventCount - left;
      return { text: `事故疑いクリップ ${h.eventCount}件${deleted > 0 ? ` · ${deleted}件削除済み` : ""}` };
    }
    if (h.status === "FAILURE") return { text: h.errorMessage || "分析中にエラーが発生しました", cls: "error" };
    if (h.status === "CANCELLED") return { text: "ユーザー要求によりキャンセル", cls: "muted" };
    return { text: progressText(h.taskId) || "分析進行中", cls: "active" };
  };
  const hasActiveAnalysis = history.some((h) => h.status === "PENDING" || h.status === "PROCESSING");

  return (
    <aside className="analysis-history">
      <div className="analysis-history-header">
        <h3>分析履歴</h3>
        <span className="analysis-history-count">計 {history.length}回</span>
        <button
          className="analysis-history-clear"
          type="button"
          disabled={history.length === 0 || hasActiveAnalysis || clearingHistory}
          title={hasActiveAnalysis ? "分析完了後に初期化できます" : "この動画の分析履歴と事故クリップを初期化"}
          onClick={onClearHistory}
        >
          {clearingHistory ? "初期化中…" : "初期化"}
        </button>
      </div>
      {history.length === 0 ? (
        <div className="event-empty-hint">まだ分析履歴がありません。</div>
      ) : (
        <ol className="analysis-history-list">
          {[...history].reverse().map((h) => {
            const st = HISTORY_STATUS[h.status] || { label: h.status, cls: "cancelled" };
            const at = formatAnalyzedAt(h.requestedAt);
            const took =
              h.requestedAt && h.finishedAt
                ? formatDuration((new Date(h.finishedAt) - new Date(h.requestedAt)) / 1000)
                : "";
            const result = resultOf(h);
            // 残存クリップがある分析のみクリックしてクリップ表示可能
            const hasClips = (remain[h.no] || 0) > 0;
            const body = (
              <>
                <div className="ahi-top">
                  <span className="ahi-no">分析 #{h.no}</span>
                  <span className={`ahi-status ${st.cls}`}>{st.label}</span>
                  {at && <span className="ahi-date">{at[0]}</span>}
                </div>
                {at && (
                  <div className="ahi-time">
                    {at[1]} リクエスト{took && ` · 所要時間: ${took}`}
                  </div>
                )}
                <div className={`ahi-result ${result.cls || ""}`} title={result.text}>
                  {result.text}
                </div>
              </>
            );
            return (
              <li className={`analysis-history-item ${hasClips && activeNo === h.no ? "active" : ""}`} key={h.taskId}>
                {hasClips ? (
                  <button className="history-video-link" onClick={() => onSelect(h)} aria-pressed={activeNo === h.no}>
                    {body}
                    <span className="history-video-action">
                      {activeNo === h.no ? "強調解除" : "クリップ一覧で強調"} <span aria-hidden="true">{activeNo === h.no ? "✕" : "←"}</span>
                    </span>
                  </button>
                ) : (
                  <div className="history-video-link is-static">{body}</div>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </aside>
  );
}

function getVideoDisplayTitle(video, allVideos) {
  if (!video) return "";
  if (video.title) return video.title; // アップロード時に入力されたタイトルがあればそのまま使用
  const sameDateVideos = (allVideos || []).filter((v) => v.date === video.date);
  if (sameDateVideos.length > 1) {
    const sorted = [...sameDateVideos].sort((a, b) => a.id - b.id);
    const index = sorted.findIndex((v) => v.id === video.id);
    const num = index >= 0 ? index + 1 : 1;
    return `${video.date} 録画データ #${num}`;
  }
  return `${video.date} 録画データ`;
}

function formatMonthLabel(date) {
  return `${date.getFullYear()}年 ${date.getMonth() + 1}月`;
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

function groupVideosByDate(videos) {
  return videos.reduce((groups, video) => {
    groups[video.date] = [...(groups[video.date] ?? []), video];
    return groups;
  }, {});
}

// ダッシュボードコンポーネント
function Dashboard({ onLogout, view, currentUser, onUpdateUser, isDarkMode, setIsDarkMode }) {
  const navigate = useNavigate();
  const { videoId } = useParams();

  // 状態管理
  const [videos, setVideos] = useState([]); // APIから読み込んだ動画一覧
  const [isLoadingVideos, setIsLoadingVideos] = useState(true);
  const [videoLoadError, setVideoLoadError] = useState(null);
  const [filterDays, setFilterDays] = useState(9999); // デフォルト全期間
  const [searchQuery, setSearchQuery] = useState(""); // 検索ワード

  const [selectedVideo, setSelectedVideo] = useState(null); // nullなら一覧(グリッド)画面、値があれば動画再生画面
  const [currentEventId, setCurrentEventId] = useState(null); // 選択中のイベントマーカー
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0); // 実際の動画再生位置(秒)
  const [selectedCalendarDate, setSelectedCalendarDate] = useState(null); // カレンダーで選択した日付
  const [datePickerVideos, setDatePickerVideos] = useState(null); // 同一日付の複数動画選択モーダル用
  const [playbackSpeed, setPlaybackSpeed] = useState("1");
  const [volume, setVolume] = useState(70);
  const [isMuted, setIsMuted] = useState(true);
  const [showVideoTapControls, setShowVideoTapControls] = useState(false);
  const [videoClickFeedback, setVideoClickFeedback] = useState(null);
  const [mediaDuration, setMediaDuration] = useState(0); // <video>の実際の長さ(秒、小数) — シークバー計算用
  const [isSeeking, setIsSeeking] = useState(false); // シークバードラッグ中フラグ
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isViewportFullscreen, setIsViewportFullscreen] = useState(false);
  const [isTheaterMode, setIsTheaterMode] = useState(false);
  const [isHistoryOpen, setIsHistoryOpen] = useState(true); // 右側パネル(進捗ステップ+分析履歴)
  const [highlightAnalysisNo, setHighlightAnalysisNo] = useState(null); // 履歴で選択した分析番号(左側リストで強調)
  const [clearingHistory, setClearingHistory] = useState(false);
  const [clipView, setClipView] = useState("cam");
  const [isPlayerSettingsOpen, setIsPlayerSettingsOpen] = useState(false);
  const camChipRef = useRef(null);
  const camOverlayChipRef = useRef(null);
  const playerSettingsRef = useRef(null);
  const videoTapControlsTimerRef = useRef(null);
  const videoClickFeedbackTimerRef = useRef(null);
  const videoClickFeedbackIdRef = useRef(0);
  const eventListRef = useRef(null);
  const [calendarMonth, setCalendarMonth] = useState(new Date("2026-05-01T00:00:00"));
  const playerContainerRef = useRef(null);
  const fullscreenStartRectRef = useRef(null);
  const fullscreenRectRef = useRef(null);
  const viewportFullscreenStartRectRef = useRef(null);
  const viewportFullscreenRef = useRef(false);
  const nativeFullscreenRef = useRef(false);
  const seekHoverRef = useRef(null); // シークバー上のツールチップ要素
  const isSeekingRef = useRef(false);
  const seekStartXRef = useRef(0);
  const seekDragThresholdRef = useRef(3);
  const didDragSeekRef = useRef(false);
  const seekCaptureTargetRef = useRef(null);
  const seekInitialTimeRef = useRef(0);
  const seekWasPlayingRef = useRef(false);
  const pendingSeekTimeRef = useRef(null);
  const lastSeekTargetRef = useRef(null);
  const seekPreviewFrameRef = useRef(0);
  const seekPreviewTimerRef = useRef(null);
  const seekPreviewLastAppliedAtRef = useRef(0);
  const seekPointerTypeRef = useRef(null);
  const detectAbortControllerRef = useRef(null);
  const analyzeAbortControllerRef = useRef(null);

  // バウンディングボックス / 車両指定モード状態 ("none" | "manual" | "auto")
  const [selectionMode, setSelectionMode] = useState("none");
  const [isBBoxMode, setIsBBoxMode] = useState(false);

  const [bboxList, setBboxList] = useState([]);
  const [currentDraw, setCurrentDraw] = useState(null); // { startX, startY, endX, endY }
  const [isDrawing, setIsDrawing] = useState(false);
  const bboxOverlayRef = useRef(null);
  const videoElRef = useRef(null); // <video> 要素 (原本解像度換算用)
  const videoPointerGestureRef = useRef(null);
  const bboxDeletePointerRef = useRef(null);

  // YOLO 車両自動検出およびマウスHover関連状態
  const [isDetectingVehicles, setIsDetectingVehicles] = useState(false);
  const [detectedBoxes, setDetectedBoxes] = useState([]); // [{ id, class_name, confidence, bbox: [x1,y1,x2,y2] }]
  const [hoveredDetectedBox, setHoveredDetectedBox] = useState(null);


  // 事故検知実行 / 分析関連状態
  const [showAnalyzeConfirm, setShowAnalyzeConfirm] = useState(false); // 確認ポップオーバー
  const [toast, setToast] = useState(null); // { type, message }
  // 同時分析タスク一覧: [{ taskId, videoId, dateLabel, estimatedSec, startedAt }]
  const [analyzingJobs, setAnalyzingJobs] = useState([]);
  const [showJobsDropdown, setShowJobsDropdown] = useState(false);
  const [now, setNow] = useState(() => Date.now()); // 進捗計算用の現在時刻(0.5sごとに更新)
  const selectedVideoIdRef = useRef(null); // ポーリング中の最新選択動画ID
  const aliveRef = useRef(true); // 画面マウント状態
  const pollingRef = useRef(new Set()); // ポーリング中タスクIDセット
  const [clipEvent, setClipEvent] = useState(null); // 事故クリップポップアップ対象
  const [pendingDeleteEvent, setPendingDeleteEvent] = useState(null); // 削除確認ダイアログ対象
  const [pendingClearAll, setPendingClearAll] = useState(false); // 一括削除確認ダイアログ
  const [infoVideo, setInfoVideo] = useState(null); // 動画情報ポップアップ対象

  // アップロードモーダル
  const [showUpload, setShowUpload] = useState(false);

  // 動画削除モード
  const [deleteMode, setDeleteMode] = useState(false);
  const [selectedForDelete, setSelectedForDelete] = useState([]); // 削除対象動画IDリスト

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
      showToast("削除する動画を選択してください。", "warning");
      return;
    }
    if (!window.confirm(`選択した${selectedForDelete.length}件の動画を削除しますか？元に戻すことはできません。`)) {
      return;
    }
    try {
      await Promise.all(selectedForDelete.map((id) => api.deleteVideo(id)));
      showToast(`${selectedForDelete.length}件の動画を削除しました。`, "success");
      exitDeleteMode();
      await loadVideos();
    } catch (e) {
      showToast(`削除失敗: ${e.message}`, "error");
    }
  };

  // 下部通知。通知ごとに番号を付与し、タイマーは自身の通知が表示中の場合のみ閉じる
  // (先行通知のタイマーが後続の通知 — 例: 7秒間の分析結果 — を早期に閉じるのを防止)
  const showToast = useCallback((message, type = "warning", durationMs = 2600) => {
    const id = ++toastSeq;
    setToast({ id, message, type });
    setTimeout(() => setToast((t) => (t && t.id === id ? null : t)), durationMs);
  }, []);

  // プロフィール & 設定関連状態
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [activeSettingsTab, setActiveSettingsTab] = useState("account"); // "account", "security"

  // 設定情報 (現在のログインユーザー情報と同期)
  const currentUsername = currentUser?.username || "user";
  const [profileName, setProfileName] = useState(currentUser?.name || "");
  const [profilePhone, setProfilePhone] = useState(currentUser?.phone || "");
  const [profileEmail, setProfileEmail] = useState(currentUser?.email || "");

  const openSettings = () => {
    setProfileName(currentUser?.name || "");
    setProfilePhone(currentUser?.phone || "");
    setProfileEmail(currentUser?.email || "");
    setIsSettingsOpen(true);
  };

  const displayName = currentUser?.name || currentUser?.username || "ユーザー";
  const avatarInitial = (displayName[0] || currentUsername[0] || "U").toUpperCase();
  const roleLabel = currentUser?.role === "ADMIN" ? "システム管理者" : "一般ユーザー";

  // パスワード変更フィールド
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  // APIから動画一覧をロード
  const loadVideos = async () => {
    setIsLoadingVideos(true);
    setVideoLoadError(null);
    try {
      const list = await api.listVideos(9999);
      setVideos(list);
      return list;
    } catch (e) {
      const message = `動画一覧を読み込めませんでした: ${e.message}`;
      setVideoLoadError(message);
      showToast(message, "error");
      return [];
    } finally {
      setIsLoadingVideos(false);
    }
  };

  useEffect(() => {
    let active = true;
    api.listVideos(9999)
      .then((list) => {
        if (active) setVideos(list);
      })
      .catch((error) => {
        if (active) setVideoLoadError(`動画一覧を読み込めませんでした: ${error.message}`);
      })
      .finally(() => {
        if (active) setIsLoadingVideos(false);
      });
    return () => { active = false; };
  }, []);

  // ルート(videoId)に合わせてselectedVideoを設定
  useEffect(() => {
    if (view !== "watch" || !videoId) {
      // watch画面でなければ選択動画を解除 (ルート同期ガード)
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setSelectedVideo(null);
      return;
    }
    // リストのスクロール位置が残らないよう詳細画面遷移時に即座に最上部へスクロール。
    window.scrollTo(0, 0);
    let cancelled = false;
    (async () => {
      try {
        const v = await api.getVideo(videoId);
        if (cancelled) return;
        setSelectedVideo(v);
        setCurrentEventId(null);
        setIsPlaying(false);
        setIsTheaterMode(false);
        setIsHistoryOpen(true);
        setHighlightAnalysisNo(null);
        setBboxList([]);
        setIsBBoxMode(false);
        setSelectionMode("none");
        setShowAnalyzeConfirm(false);

        setCurrentTime(0);
        setMediaDuration(0);
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
        showToast(`動画の取得に失敗しました: ${e.message}`, "error");
        navigate("/videos");
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, videoId]);

  // 期間フィルターのみ適用した一覧
  const periodVideos = useMemo(() => {
    if (filterDays >= 9999) return videos;
    const startDate = formatDate(addDays(new Date(), -filterDays));
    return videos.filter((video) => video.date >= startDate);
  }, [filterDays, videos]);

  // 期間 + 検索ワードフィルタリング — 動画一覧ページ用
  const filteredVideos = useMemo(() => {
    let list = periodVideos;

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      // タイトルはカード表示名('2026-10-12 録画データ #5')で検索
      list = list.filter(
        (v) =>
          (v.date || "").toLowerCase().includes(q) ||
          getVideoDisplayTitle(v, videos).toLowerCase().includes(q) ||
          (v.events && v.events.some((e) => (e.title || "").toLowerCase().includes(q)))
      );
    }

    // 録画日の新しい順にソート (同一日は後にアップロードされた動画を優先)
    return [...list].sort((a, b) => {
      if (a.date !== b.date) return (b.date || "").localeCompare(a.date || "");
      return b.id - a.id;
    });
  }, [periodVideos, searchQuery, videos]);

  // メイン画面動画リストのページネーション — 1ページ = 現在の列数 × 行数(2行以上、最小6件)。
  // 3列以下(通常画面)は従来通り6件、幅広画面や縮小表示で列が増加した場合は行を充填(例: 6列 → 12件)
  const [videoGridEl, setVideoGridEl] = useState(null);
  const gridCols = useGridColumns(videoGridEl);
  const isMobilePagination = useMediaQuery("(max-width: 640px)");
  const VIDEOS_PER_PAGE = gridCols * Math.max(2, Math.ceil(6 / gridCols));
  const [currentPage, setCurrentPage] = useState(1);

  const totalPages = Math.max(1, Math.ceil(filteredVideos.length / VIDEOS_PER_PAGE));
  const visiblePage = Math.min(currentPage, totalPages);

  // 1ページあたりの表示数が変更された場合
  const prevPerPage = useRef(VIDEOS_PER_PAGE);
  useEffect(() => {
    if (prevPerPage.current === VIDEOS_PER_PAGE) return;
    const firstIndex = (visiblePage - 1) * prevPerPage.current;
    prevPerPage.current = VIDEOS_PER_PAGE;
    setCurrentPage(Math.floor(firstIndex / VIDEOS_PER_PAGE) + 1);
  }, [VIDEOS_PER_PAGE, visiblePage]);

  const paginatedVideos = useMemo(() => {
    const startIndex = (visiblePage - 1) * VIDEOS_PER_PAGE;
    return filteredVideos.slice(startIndex, startIndex + VIDEOS_PER_PAGE);
  }, [filteredVideos, visiblePage, VIDEOS_PER_PAGE]);

  const handlePageChange = (newPage) => {
    setCurrentPage(newPage);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const getPageNumbers = () => {
    const maxButtons = isMobilePagination ? 3 : 5;
    let start = Math.max(1, visiblePage - Math.floor(maxButtons / 2));
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
    return groupVideosByDate(videos);
  }, [videos]);

  const calendarDays = useMemo(() => getCalendarDays(calendarMonth), [calendarMonth]);

  const moveCalendarMonth = (offset) => {
    setCalendarMonth((month) => new Date(month.getFullYear(), month.getMonth() + offset, 1));
  };

  const handleCalendarDateClick = (date) => {
    const dateText = formatDate(date);
    setSelectedCalendarDate(dateText);
    const dayVideos = videosByDate[dateText] ?? [];
    if (dayVideos.length === 1) {
      handleWatchVideo(dayVideos[0]);
    } else if (dayVideos.length > 1) {
      setDatePickerVideos({ dateText, videos: dayVideos });
    }
  };

  // 動画をクリックして再生モードへ遷移
  const handleWatchVideo = (video) => {
    navigate(`/videos/${video.id}`);
  };

  // 一覧へ戻る
  const handleBackToHome = () => {
    navigate("/videos");
  };

  // 特定の事故イベント1件削除
  // ゴミ箱 → 即時削除せず画面中央に確認ダイアログを表示(サーバー側でもクリップファイルを削除)
  const requestDeleteEvent = (event, e) => {
    if (e) e.stopPropagation();
    setPendingDeleteEvent(event);
  };

  const handleDeleteEvent = async (eventId) => {
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
      showToast("検知イベントが削除されました。", "info");
    } catch (err) {
      showToast(`イベント削除失敗: ${err.message}`, "error");
    }
  };

  // 動画のすべての事故イベントの一括削除要求
  const requestClearAllEvents = () => {
    if (!selectedVideo || selectedVideo.events.length === 0) return;
    setPendingClearAll(true);
  };

  const handleClearAllEvents = async () => {
    if (!selectedVideo || selectedVideo.events.length === 0) return;
    try {
      await api.clearEvents(selectedVideo.id);
      setSelectedVideo((prev) => (prev ? { ...prev, events: [] } : null));
      setVideos((prev) =>
        prev.map((v) => (v.id === selectedVideo.id ? { ...v, events: [] } : v))
      );
      setCurrentEventId(null);
      showToast("すべての検知イベントが削除されました。", "info");
    } catch (err) {
      showToast(`イベント一括削除失敗: ${err.message}`, "error");
    }
  };

  const handleClearAnalysisHistory = async () => {
    if (!selectedVideo || clearingHistory || selectedVideo.analysisHistory?.length === 0) return;
    const historyCount = selectedVideo.analysisHistory?.length || 0;
    if (!window.confirm(`この動画の分析履歴 ${historyCount}回を初期化しますか？事故クリップも削除されますが元動画は保持されます。`)) return;
    setClearingHistory(true);
    try {
      await api.clearAnalysisHistory(selectedVideo.id);
      setSelectedVideo((current) => current?.id === selectedVideo.id
        ? { ...current, analysisHistory: [], events: [], analyzedBefore: false, lastTask: null }
        : current);
      setVideos((current) => current.map((video) => video.id === selectedVideo.id
        ? { ...video, events: [], analyzedBefore: false, lastTask: null }
        : video));
      setCurrentEventId(null);
      setHighlightAnalysisNo(null);
      showToast("この動画の分析履歴を初期化しました。", "success");
    } catch (err) {
      showToast(`分析履歴の初期化に失敗しました: ${err.message}`, "error");
    } finally {
      setClearingHistory(false);
    }
  };


  // 手動による車両指定モード開始
  const handleStartManualDesignate = () => {
    // 車両検知と同様に動画を一時停止し、現在表示中のフレーム上で車両を描画させる
    if (videoElRef.current) {
      videoElRef.current.pause();
      setIsPlaying(false);
    }
    setSelectionMode("manual");
    setIsBBoxMode(true);
    setHoveredDetectedBox(null);
    // ガイド文言は一時的なトースト通知ではなくボタン上のツールチップとしてモード終了まで常時表示 (.designate-hint)
  };

  // 検出/指定モード解除
  const handleEndDesignateMode = () => {
    setSelectionMode("none");
    setIsBBoxMode(false);
    setCurrentDraw(null);
    setIsDrawing(false);
    setHoveredDetectedBox(null);
  };

  // YOLO 車両検出キャンセル
  const handleCancelDetectVehicles = () => {
    if (detectAbortControllerRef.current) {
      detectAbortControllerRef.current.abort();
      detectAbortControllerRef.current = null;
    }
    setIsDetectingVehicles(false);
    setSelectionMode("none");
    showToast("車両検出がキャンセルされました。", "info");
  };

  // YOLO 車両自動検出実行
  const handleAutoDetectVehicles = async () => {
    if (!selectedVideo) return;
    if (videoElRef.current) {
      videoElRef.current.pause();
      setIsPlaying(false);
    }
    setSelectionMode("auto");
    setIsDetectingVehicles(true);

    const controller = new AbortController();
    detectAbortControllerRef.current = controller;

    try {
      const request = await api.detectVehicles(selectedVideo.id, currentTime, controller.signal);
      let result;
      while (!result) {
        if (controller.signal.aborted) {
          throw new DOMException("車両検知がキャンセルされました。", "AbortError");
        }
        const status = await api.getVehicleDetectionTask(
          selectedVideo.id, request.task_id, controller.signal,
        );
        if (status.status === "SUCCESS") {
          result = status;
          break;
        }
        if (status.status === "FAILURE" || status.status === "CANCELLED") {
          throw new Error(status.error_message || "車両検知に失敗しました。");
        }
        await new Promise((resolve) => setTimeout(resolve, 700));
      }
      const list = result.detected_vehicles || [];
      setDetectedBoxes(list);
      setIsBBoxMode(true);
      // 検知成功時は通知を表示しない(動画上に車両が即座に描画されるため)。1台も検出されなかった場合のみ次のアクションを案内。
      if (list.length === 0) {
        showToast("検出された車両がありません。手動による車両指定をご利用ください。", "warning");
      }
    } catch (err) {
      if (err.name === "AbortError") {
        return;
      }
      showToast(`車両検出失敗: ${err.message}`, "error");
    } finally {
      setIsDetectingVehicles(false);
      detectAbortControllerRef.current = null;
    }
  };



  // 「事故区間を検出」クリック → 車両ボックスがなければ警告、あれば確認ポップオーバー
  const handleAnalyzeClick = () => {
    if (bboxList.length === 0) {
      showToast("事故検知対象の車両を指定してください。", "warning");
      return;
    }
    setShowAnalyzeConfirm(true);
  };

  // 事故検知分析タスクのキャンセル
  // 閲覧中動画の詳細(分析履歴含む)を再取得 — その間に別動画へ移動した場合は上書きしない
  const refreshSelectedVideo = async (videoId) => {
    if (videoId == null) return;
    try {
      const v = await api.getVideo(videoId);
      setSelectedVideo((current) => current?.id === videoId ? v : current);
    } catch {
      /* ignore */
    }
  };

  const handleCancelAnalysis = async (taskId) => {
    if (taskId) {
      try {
        const cancelledVideoId = analyzingJobs.find((j) => j.taskId === taskId)?.videoId;
        await api.cancelTask(taskId);
        removeJob(taskId);
        showToast("事故検知の分析タスクをキャンセルしました。", "info");
        refreshSelectedVideo(cancelledVideoId);
        loadVideos();
      } catch (e) {
        showToast(`分析キャンセル失敗: ${e.message}`, "error");
      }
    } else {
      showToast("分析リクエストがキャンセルされました。", "info");
    }
  };

  // 分析タスク状態ポーリング
  const removeJob = (taskId) =>
    setAnalyzingJobs((prev) => prev.filter((j) => j.taskId !== taskId));

  // 分析完了通知
  const notifyDone = (title, body, type) => {
    showToast(`${title}: ${body}`, type, 7000);
    try {
      if (document.hidden && "Notification" in window && Notification.permission === "granted") {
        new Notification(title, { body, icon: "/siot-logo.svg" });
      }
    } catch {
      /* ignore */
    }
  };

  const pollTask = (taskId, videoId) => {
    if (pollingRef.current.has(taskId)) return;
    pollingRef.current.add(taskId);
    const finish = () => {
      pollingRef.current.delete(taskId);
      removeJob(taskId);
    };
    const poll = async () => {
      if (!aliveRef.current) return;
      try {
        const status = await api.getTaskStatus(taskId);
        if (!aliveRef.current) return;
        if (status.status === "SUCCESS") {
          finish();
          notifyDone("分析完了", `事故疑惑区間 ${status.events.length}件`, "success");
          // 分析対象の動画を現在も閲覧中の場合のみ画面更新 (別動画へ移動済みの場合は上書きしない)
          if (selectedVideoIdRef.current === videoId) {
            const v = await api.getVideo(videoId);
            setSelectedVideo(v);
          }
          loadVideos();
          return;
        }
        if (status.status === "FAILURE") {
          finish();
          notifyDone("分析失敗", status.error_message || "エラー", "error");
          if (selectedVideoIdRef.current === videoId) {
            const v = await api.getVideo(videoId);
            setSelectedVideo(v);
          }
          return;
        }
        if (status.status === "CANCELLED") {
          finish();
          showToast("事故検知の分析がキャンセルされました。", "info");
          refreshSelectedVideo(videoId);
          return;
        }
        // 進捗度(%)とサーバーが実測した速度基準の残り時間(秒)を反映。
        // 取得時刻(etaAt)を保持し次回確認まで画面上で1秒ずつ減算カウントダウン。
        const receivedAt = Date.now();
        setAnalyzingJobs((prev) =>
          prev.map((j) => {
            if (j.taskId !== taskId) return j;
            const next = { ...j, status: status.status };
            if (typeof status.progress === "number") {
              next.progress = status.progress;
              next.progressAt = receivedAt;
            }
            next.etaSec = status.eta_sec ?? null;
            next.etaMin = !!status.eta_is_min;
            next.etaAt = receivedAt;
            next.phase = status.stage ?? null;
            next.phaseCount = status.stage_count ?? null;
            next.phaseIndex = status.stage_index ?? null;
            return next;
          }));
        setTimeout(poll, 2000);
      } catch (e) {
        finish();
        showToast(`ステータス取得失敗: ${e.message}`, "error");
      }
    };
    poll();
  };

  // 失敗・中断された直近の分析を同一車両ボックスで再実行
  const handleRetryAnalysis = async () => {
    const last = selectedVideo?.lastTask;
    if (!last) return;
    const videoIdForJob = selectedVideo.id;
    try {
      const res = await api.retryTask(last.id);
      const job = {
        taskId: res.task_id,
        videoId: videoIdForJob,
        dateLabel: selectedVideo.date,
        status: res.status,
      };
      showToast(res.already_running ? "この動画は既に分析中です。" : "分析を再開しました。", "info");
      setAnalyzingJobs((prev) => (prev.some((j) => j.taskId === res.task_id) ? prev : [...prev, job]));
      pollTask(res.task_id, videoIdForJob);
      setSelectedVideo((v) =>
        v ? { ...v, lastTask: { id: res.task_id, status: res.status, errorMessage: null } } : v);
      refreshSelectedVideo(videoIdForJob);
    } catch (e) {
      showToast(`再実行失敗: ${e.message}`, "error");
    }
  };

  // 画面マウント時(リロード・再接続含む)にサーバー上に残存する自身の分析を復元して継続追跡。
  // 分析リストはブラウザメモリのみに保持されるため、これがないとウィンドウを閉じて再度開いた際に実行中の
  // 分析が消失したように見え同一動画を再リクエストしてしまう。
  useEffect(() => {
    aliveRef.current = true;
    (async () => {
      try {
        const tasks = await api.listActiveTasks();
        if (!aliveRef.current) return;
        tasks.forEach((t) => {
          const job = {
            taskId: t.task_id,
            videoId: t.video_id,
            dateLabel: t.recording_date || (t.created_at || "").slice(0, 10),
            startedAt: Date.now(),
            status: t.status,
            progress: t.progress ?? undefined,
            etaSec: t.eta_sec ?? null,
            etaMin: !!t.eta_is_min,
            etaAt: Date.now(),
            phase: t.stage ?? null,
            phaseCount: t.stage_count ?? null,
            phaseIndex: t.stage_index ?? null,
            progressAt: Date.now(),
          };
          setAnalyzingJobs((prev) => (prev.some((j) => j.taskId === t.task_id) ? prev : [...prev, job]));
          pollTask(t.task_id, t.video_id);
        });
      } catch {
        /* ignore */
      }
    })();
    return () => {
      aliveRef.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 確認ポップオーバーから「実行」 → 分析リクエスト
  const runAnalysis = async () => {
    setShowAnalyzeConfirm(false);
    const box = bboxList[0];
    if (!box) return;

    // bboxList は元動画の解像度座標 — 現在の画面サイズに関わらずそのまま送信
    const bbox = {
      bbox_xmin: box.xmin,
      bbox_ymin: box.ymin,
      bbox_xmax: box.xmax,
      bbox_ymax: box.ymax,
    };

    // 分析開始: 指定モードを終了し該当動画を「分析中」として表示
    // (描画領域のみ無効化するとモード状態が残り分析中も検知ボタンがアクティブに見えてしまうため)
    const videoId = selectedVideo.id;
    handleEndDesignateMode();

    // 予想所要時間(秒)の推定: スライディングウィンドウ数(総フレーム/stride)基準 (CPU ~1s/ウィンドウ)
    // ※ 分母はバックエンド config.PREDICT_WINDOW_STRIDE(現在 5)と必ず一致させること
    const totalFrames = Math.round(
      (selectedVideo.duration || 0) * (selectedVideo.fps || 30));
    const estimatedSec = Math.max(10, Math.round((totalFrames / 5) * 1.1));
    const job = {
      taskId: null,
      videoId,
      dateLabel: selectedVideo.date,
      estimatedSec,
      startedAt: Date.now(),
    };

    // 分析完了時に他タブ閲覧中であっても通知できるよう初回のみ通知権限をリクエスト
    try {
      if ("Notification" in window && Notification.permission === "default") {
        const r = Notification.requestPermission();
        if (r && r.catch) r.catch(() => {});
      }
    } catch {
      /* ignore */
    }

    const controller = new AbortController();
    analyzeAbortControllerRef.current = controller;

    try {
      const res = await api.analyze(videoId, bbox, controller.signal);
      job.taskId = res.task_id;
      job.status = res.status;
      if (res.already_running) {
        // 同一動画に対して既に待機・実行中の分析が存在する場合、サーバーは新規作成せず該当タスクを返却
        showToast("この動画は既に分析中です。進行状況を表示します。", "info");
      }
      setAnalyzingJobs((prev) => (prev.some((j) => j.taskId === res.task_id) ? prev : [...prev, job]));
      pollTask(res.task_id, videoId);
      refreshSelectedVideo(videoId);
    } catch (e) {
      if (e.name === "AbortError") {
        return;
      }
      showToast(`分析リクエスト失敗: ${e.message}`, "error");
    } finally {
      analyzeAbortControllerRef.current = null;
    }
  };

  const exitViewportFullscreen = useCallback(() => {
    viewportFullscreenRef.current = false;
    setIsViewportFullscreen(false);
    setIsFullscreen(false);
  }, []);

  const handleToggleFullscreen = useCallback(() => {
    const player = playerContainerRef.current;
    if (!player) return;

    if (viewportFullscreenRef.current) {
      exitViewportFullscreen();
      return;
    }

    if (document.fullscreenElement || document.webkitFullscreenElement || document.webkitCurrentFullScreenElement) {
      fullscreenRectRef.current = player.getBoundingClientRect();
      if (document.exitFullscreen) document.exitFullscreen().catch(() => {});
      else document.webkitExitFullscreen?.();
      return;
    }

    const enterViewportFullscreen = () => {
      if (playerContainerRef.current !== player || !player.isConnected) return;
      viewportFullscreenStartRectRef.current = fullscreenStartRectRef.current || player.getBoundingClientRect();
      fullscreenStartRectRef.current = null;
      viewportFullscreenRef.current = true;
      setIsViewportFullscreen(true);
      setIsFullscreen(true);
    };

    const requestFullscreen = document.fullscreenEnabled !== false && player.requestFullscreen
      ? player.requestFullscreen
      : document.webkitFullscreenEnabled && player.webkitRequestFullscreen;
    if (!requestFullscreen) {
      enterViewportFullscreen();
      return;
    }

    fullscreenStartRectRef.current = player.getBoundingClientRect();
    try {
      const request = requestFullscreen.call(player);
      request?.catch?.(enterViewportFullscreen);
    } catch {
      enterViewportFullscreen();
    }
  }, [exitViewportFullscreen]);

  useLayoutEffect(() => {
    if (!isViewportFullscreen) return undefined;
    const fromRect = viewportFullscreenStartRectRef.current;
    viewportFullscreenStartRectRef.current = null;
    const player = playerContainerRef.current;
    if (!player || !fromRect || prefersReducedMotion() || typeof player.animate !== "function") return undefined;
    if (!window.matchMedia("(max-width: 720px), (pointer: coarse)").matches) return undefined;

    const toRect = player.getBoundingClientRect();
    if (!fromRect.width || !fromRect.height || !toRect.width || !toRect.height) return undefined;
    const x = fromRect.left + fromRect.width / 2 - (toRect.left + toRect.width / 2);
    const y = fromRect.top + fromRect.height / 2 - (toRect.top + toRect.height / 2);
    const scale = Math.max(0.85, Math.min(1, fromRect.width / toRect.width));
    const animation = player.animate([
      { opacity: 0, transformOrigin: "center", transform: `translate3d(${x}px, ${y}px, 0) scale(${scale})` },
      { opacity: 1, transformOrigin: "center", transform: "translate3d(0, 0, 0) scale(1)" },
    ], { duration: 360, easing: "cubic-bezier(0.22, 1, 0.36, 1)" });
    return () => animation.cancel();
  }, [isViewportFullscreen]);

  useEffect(() => {
    if (!isViewportFullscreen) return undefined;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKeyDown = (event) => {
      if (event.key === "Escape") exitViewportFullscreen();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [isViewportFullscreen, exitViewportFullscreen]);

  useEffect(() => {
    if (!selectedVideo && viewportFullscreenRef.current) exitViewportFullscreen();
  }, [selectedVideo, exitViewportFullscreen]);

  useEffect(() => {
    if (!isTheaterMode || !window.matchMedia("(max-width: 720px)").matches) return undefined;
    const frame = window.requestAnimationFrame(() => {
      window.scrollTo({ top: 0, left: 0, behavior: prefersReducedMotion() ? "auto" : "smooth" });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [isTheaterMode]);

  // ── 実際の <video> 再生制御 ──────────────────────────────
  // 車両検出中は再生・シークを制限
  const togglePlay = useCallback(() => {
    const v = videoElRef.current;
    if (!v || isDetectingVehicles) return;
    if (v.paused) v.play();
    else v.pause();
  }, [isDetectingVehicles]);

  useEffect(() => () => {
    clearTimeout(videoTapControlsTimerRef.current);
    clearTimeout(videoClickFeedbackTimerRef.current);
    clearTimeout(seekPreviewTimerRef.current);
    if (seekPreviewFrameRef.current) cancelAnimationFrame(seekPreviewFrameRef.current);
  }, []);

  const revealVideoTapControls = () => {
    setShowVideoTapControls(true);
    clearTimeout(videoTapControlsTimerRef.current);
    videoTapControlsTimerRef.current = setTimeout(() => setShowVideoTapControls(false), 2600);
  };

  const toggleVideoTapControls = () => {
    if (showVideoTapControls) {
      clearTimeout(videoTapControlsTimerRef.current);
      setShowVideoTapControls(false);
      return;
    }
    revealVideoTapControls();
  };

  const dismissVideoTapControls = useCallback(() => {
    clearTimeout(videoTapControlsTimerRef.current);
    setShowVideoTapControls(false);
  }, []);

  const usesTouchVideoControls = () =>
    window.matchMedia("(pointer: coarse)").matches ||
    window.matchMedia("(max-width: 1024px)").matches;

  const showPlayerFeedback = useCallback((action) => {
    if (usesTouchVideoControls()) return;
    setVideoClickFeedback({ action, id: ++videoClickFeedbackIdRef.current });
    clearTimeout(videoClickFeedbackTimerRef.current);
    videoClickFeedbackTimerRef.current = setTimeout(() => setVideoClickFeedback(null), 820);
  }, []);

  const toggleDesktopVideoPlayback = useCallback(() => {
    const video = videoElRef.current;
    if (!video || isDetectingVehicles) return;
    const action = video.paused ? "play" : "pause";
    showPlayerFeedback(action);
    dismissVideoTapControls();
    togglePlay();
  }, [dismissVideoTapControls, isDetectingVehicles, showPlayerFeedback, togglePlay]);

  // 再生バーの長さ基準
  const playerDuration = mediaDuration || selectedVideo?.duration || 0;
  const seekBySeconds = (delta) => {
    const video = videoElRef.current;
    if (!video || isDetectingVehicles) return;
    const duration = Number.isFinite(video.duration) && video.duration > 0
      ? video.duration : playerDuration;
    const nextTime = Math.max(0, Math.min(duration || Infinity, video.currentTime + delta));
    const wasPlaying = !video.paused;
    video.currentTime = nextTime;
    setCurrentTime(nextTime);
    showPlayerFeedback(delta < 0 ? "replay5" : "forward5");
    if (wasPlaying && video.paused) video.play().catch(() => {});
    revealVideoTapControls();
  };
  const isCurrentTimeInEvent = selectedVideo?.events?.some((event) => {
    const start = event.startSec ?? event.timestamp;
    const end = Math.max(start, event.endSec ?? event.endTimestamp ?? start);
    return currentTime >= start && currentTime <= end;
  });

  // 再生バー: クリックまたはドラッグでシーク、ホバー時に時刻をツールチップ表示
  const seekRatioAt = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    if (rect.width <= 0) return 0;
    return Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
  };

  const seekDurationFor = (video) =>
    Number.isFinite(video?.duration) && video.duration > 0 ? video.duration : playerDuration;

  const queueSeekPreview = (time) => {
    if (!Number.isFinite(time)) return;
    pendingSeekTimeRef.current = time;
    if (videoElRef.current?.seeking) return;
    if (seekPreviewFrameRef.current || seekPreviewTimerRef.current) return;
    const delay = seekPointerTypeRef.current === "touch"
      ? Math.max(0, 150 - (performance.now() - seekPreviewLastAppliedAtRef.current))
      : 0;
    if (delay > 0) {
      seekPreviewTimerRef.current = setTimeout(() => {
        seekPreviewTimerRef.current = null;
        if (isSeekingRef.current && pendingSeekTimeRef.current != null) {
          queueSeekPreview(pendingSeekTimeRef.current);
        }
      }, delay);
      return;
    }
    seekPreviewFrameRef.current = requestAnimationFrame(() => {
      seekPreviewFrameRef.current = 0;
      const v = videoElRef.current;
      if (!isSeekingRef.current || !v || v.seeking || pendingSeekTimeRef.current == null) return;
      const nextTime = pendingSeekTimeRef.current;
      pendingSeekTimeRef.current = null;
      if (Math.abs(v.currentTime - nextTime) < 0.001) return;
      // fastSeekはキーフレームに合わせて秒単位でスキップ可能なため、ドラッグ位置をそのまま要求。
      seekPreviewLastAppliedAtRef.current = performance.now();
      v.currentTime = nextTime;
    });
  };

  const updateSeekBubble = (e, ratio) => {
    const bubble = seekHoverRef.current;
    if (!bubble) return;
    bubble.textContent = formatTime(Math.floor(ratio * playerDuration));
    // 両端でツールチップが再生バー外にはみ出さないよう内側にクランプ
    const width = e.currentTarget.getBoundingClientRect().width;
    const half = bubble.offsetWidth / 2;
    bubble.style.left = `${Math.min(width - half, Math.max(half, ratio * width))}px`;
  };

  const handleSeekStart = (e) => {
    if (e.button !== 0 || isDetectingVehicles) return;
    const captureTarget = e.target.closest?.(".event-range") || e.currentTarget;
    captureTarget.setPointerCapture?.(e.pointerId);
    seekCaptureTargetRef.current = captureTarget;
    seekInitialTimeRef.current = videoElRef.current?.currentTime || 0;
    seekWasPlayingRef.current = !!videoElRef.current && !videoElRef.current.paused;
    seekStartXRef.current = e.clientX;
    seekDragThresholdRef.current = e.pointerType === "touch" ? 10 : 3;
    seekPointerTypeRef.current = e.pointerType;
    seekPreviewLastAppliedAtRef.current = -Infinity;
    didDragSeekRef.current = false;
    isSeekingRef.current = true;
    setIsSeeking(true);
    const ratio = seekRatioAt(e);
    const duration = seekDurationFor(videoElRef.current);
    lastSeekTargetRef.current = duration > 0 ? ratio * duration : null;
    if (lastSeekTargetRef.current != null) {
      setCurrentTime(lastSeekTargetRef.current);
    }
    updateSeekBubble(e, ratio);
  };

  const handleSeekMove = (e) => {
    if (isSeekingRef.current && Math.abs(e.clientX - seekStartXRef.current) > seekDragThresholdRef.current) {
      didDragSeekRef.current = true;
    }
    const ratio = seekRatioAt(e);
    updateSeekBubble(e, ratio);
    if (isSeekingRef.current && didDragSeekRef.current) {
      const v = videoElRef.current;
      const duration = seekDurationFor(v);
      if (v && duration > 0) {
        lastSeekTargetRef.current = ratio * duration;
        setCurrentTime(lastSeekTargetRef.current);
        // タッチ操作は最新位置のみまとめてプレビュー表示し、最終位置はリリース時に正確に適用。
        queueSeekPreview(lastSeekTargetRef.current);
      }
    }
  };

  const handleSeekEnd = (e) => {
    if (!isSeekingRef.current) return;
    isSeekingRef.current = false;
    const duration = seekDurationFor(videoElRef.current);
    const pendingTime = didDragSeekRef.current && duration > 0
      ? seekRatioAt(e) * duration
      : lastSeekTargetRef.current;
    const seekDelta = pendingTime == null ? 0 : pendingTime - seekInitialTimeRef.current;
    const shouldResumePlayback = seekWasPlayingRef.current;
    const eventId = Number(seekCaptureTargetRef.current?.dataset.eventId);
    const tappedEvent = !didDragSeekRef.current && eventId
      ? selectedVideo?.events.find((event) => event.id === eventId)
      : null;
    pendingSeekTimeRef.current = null;
    lastSeekTargetRef.current = null;
    clearTimeout(seekPreviewTimerRef.current);
    seekPreviewTimerRef.current = null;
    if (seekPreviewFrameRef.current) {
      cancelAnimationFrame(seekPreviewFrameRef.current);
      seekPreviewFrameRef.current = 0;
    }
    if (pendingTime != null && !tappedEvent) {
      const v = videoElRef.current;
      if (v) {
        v.currentTime = pendingTime;
        setCurrentTime(pendingTime);
        setCurrentEventId(null);
      }
    }
    const captureTarget = seekCaptureTargetRef.current;
    if (captureTarget?.hasPointerCapture?.(e.pointerId)) {
      captureTarget.releasePointerCapture(e.pointerId);
    }
    seekCaptureTargetRef.current = null;
    seekWasPlayingRef.current = false;
    seekPointerTypeRef.current = null;
    setIsSeeking(false);
    didDragSeekRef.current = false;
    if (Math.abs(seekDelta) > 0.05) {
      showPlayerFeedback(seekDelta < 0 ? "fastRewind" : "fastForward");
    }
    // タッチブラウザの合成clickが移動量によりキャンセルされても赤色区間のタップで再生できるようにする。
    if (tappedEvent) {
      const video = videoElRef.current;
      seekToEvent(
        tappedEvent,
        pendingTime ?? (tappedEvent.startSec ?? tappedEvent.timestamp),
        shouldResumePlayback,
      );
    } else if (shouldResumePlayback) {
      const video = videoElRef.current;
      if (video?.paused) video.play().catch(() => {});
    }
  };

  const handleSeekCancel = (e) => {
    if (!isSeekingRef.current) return;
    isSeekingRef.current = false;
    const shouldResumePlayback = seekWasPlayingRef.current;
    pendingSeekTimeRef.current = null;
    lastSeekTargetRef.current = null;
    clearTimeout(seekPreviewTimerRef.current);
    seekPreviewTimerRef.current = null;
    if (seekPreviewFrameRef.current) {
      cancelAnimationFrame(seekPreviewFrameRef.current);
      seekPreviewFrameRef.current = 0;
    }
    const captureTarget = seekCaptureTargetRef.current;
    if (captureTarget?.hasPointerCapture?.(e.pointerId)) {
      captureTarget.releasePointerCapture(e.pointerId);
    }
    seekCaptureTargetRef.current = null;
    seekWasPlayingRef.current = false;
    seekPointerTypeRef.current = null;
    didDragSeekRef.current = false;
    setIsSeeking(false);
    if (videoElRef.current) setCurrentTime(videoElRef.current.currentTime);
    if (shouldResumePlayback) {
      const video = videoElRef.current;
      if (video?.paused) video.play().catch(() => {});
    }
  };

  const handleSeeked = (e) => {
    if (!isSeekingRef.current) setCurrentTime(e.currentTarget.currentTime);
    if (isSeekingRef.current && pendingSeekTimeRef.current != null) {
      queueSeekPreview(pendingSeekTimeRef.current);
    }
  };

  useEffect(() => {
    if (!selectedVideo) return undefined;
    const onPlayerKeyDown = (e) => {
      if (e.defaultPrevented || e.isComposing || e.altKey || e.ctrlKey || e.metaKey) return;
      if (e.target.closest?.("input, textarea, select, button, a, [contenteditable='true'], [role='slider']")) return;

      const video = videoElRef.current;
      const key = e.key.toLowerCase();
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        if (!video || isDetectingVehicles) return;
        const delta = e.key === "ArrowLeft" ? -5 : 5;
        const duration = Number.isFinite(video.duration) ? video.duration : playerDuration;
        video.currentTime = Math.max(0, Math.min(duration, video.currentTime + delta));
        setCurrentTime(video.currentTime);
        showPlayerFeedback(delta < 0 ? "replay5" : "forward5");
      } else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        e.preventDefault();
        const next = Math.max(0, Math.min(100, volume + (e.key === "ArrowUp" ? 5 : -5)));
        setVolume(next);
        setIsMuted((muted) => (e.key === "ArrowUp" ? false : next === 0 ? true : muted));
        showPlayerFeedback(e.key === "ArrowUp" ? "volumeUp" : "volumeDown");
      } else if (key === "m" || e.code === "KeyM") {
        e.preventDefault();
        if (isMuted || volume === 0) {
          if (volume === 0) setVolume(50);
          setIsMuted(false);
          showPlayerFeedback("volumeUp");
        } else {
          setIsMuted(true);
          showPlayerFeedback("volumeOff");
        }
      } else if (key === "t" || e.code === "KeyT") {
        e.preventDefault();
        if (selectionMode === "none" && !isDetectingVehicles) {
          setIsTheaterMode((enabled) => !enabled);
          showPlayerFeedback("theater");
        }
      } else if (key === "f" || e.code === "KeyF") {
        e.preventDefault();
        if ((selectionMode === "none" && !isDetectingVehicles) || isFullscreen) {
          showPlayerFeedback(isFullscreen ? "fullscreenExit" : "fullscreen");
          handleToggleFullscreen();
        }
      } else if (e.code === "Space" || e.key === " ") {
        e.preventDefault();
        if (usesTouchVideoControls()) togglePlay();
        else toggleDesktopVideoPlayback();
      }
    };
    document.addEventListener("keydown", onPlayerKeyDown);
    return () => document.removeEventListener("keydown", onPlayerKeyDown);
  }, [selectedVideo, isDetectingVehicles, playerDuration, volume, isMuted, selectionMode, isFullscreen, togglePlay, toggleDesktopVideoPlayback, handleToggleFullscreen, showPlayerFeedback]);

  // イベント区間クリック → 1秒前から再生し事故直前の状況も確認できるようにする。
  const seekToEvent = (event, exactTime = null, shouldPlay = true) => {
    if (isDetectingVehicles) return;
    setCurrentEventId(event.id);
    const v = videoElRef.current;
    if (v) {
      const duration = Number.isFinite(v.duration) && v.duration > 0 ? v.duration : playerDuration;
      const eventStart = Math.max(0, (event.startSec ?? event.timestamp) - EVENT_PREROLL_SEC);
      v.currentTime = exactTime == null ? eventStart : Math.max(0, Math.min(duration, exactTime));
      setCurrentTime(v.currentTime);
      if (shouldPlay) v.play().catch(() => {});
    }
  };

  const toggleMute = () => {
    if (isMuted || volume === 0) {
      if (volume === 0) setVolume(50);
      setIsMuted(false);
      showPlayerFeedback("volumeUp");
    } else {
      setIsMuted(true);
      showPlayerFeedback("volumeOff");
    }
  };

  // 再生速度/音量を実際のvideoに反映
  useEffect(() => {
    const v = videoElRef.current;
    if (v) v.playbackRate = parseFloat(playbackSpeed);
  }, [playbackSpeed, selectedVideo]);

  useEffect(() => {
    const v = videoElRef.current;
    if (v) v.volume = volume / 100;
  }, [volume, selectedVideo]);

  useEffect(() => {
    const v = videoElRef.current;
    if (v) v.muted = isMuted;
  }, [isMuted, selectedVideo]);

  // 動画領域サイズ変更時(ウィンドウサイズ · 案内バー · エラー案内 · 全画面)に車両ボックス位置を再計算して再描画。
  // 動画領域は画面高さに合わせて縮小される可能性があるため、レンダリング時点のサイズで描画したボックスのズレを防止。
  const [overlaySize, setOverlaySize] = useState(null); // { w, h } — ボックス描画用(レンダリング中のref参照を避けるためstate管理)
  useEffect(() => {
    const el = bboxOverlayRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([entry]) => {
      // getBoundingClientRect() includes the fullscreen transition transform.
      // Measuring the visual, animated bounds can leave CAM overlays positioned
      // for an intermediate scale after fullscreen has already closed.
      const { width, height } = entry?.contentRect || el.getBoundingClientRect();
      setOverlaySize((prev) => (prev && prev.w === width && prev.h === height ? prev : { w: width, h: height }));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [selectedVideo?.id, view]);

  // 分析履歴選択時の自動スクロール
  useEffect(() => {
    if (highlightAnalysisNo == null) return;
    const el = eventListRef.current?.querySelector(".event-item.analysis-highlight");
    el?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [highlightAnalysisNo]);

  // 全画面状態の同期
  useEffect(() => {
    const onChange = () => {
      if (viewportFullscreenRef.current) return;
      const player = playerContainerRef.current;
      const fullscreenElement = document.fullscreenElement || document.webkitFullscreenElement || document.webkitCurrentFullScreenElement;
      const fullscreen = !!fullscreenElement && fullscreenElement === player;
      if (nativeFullscreenRef.current === fullscreen) return;
      nativeFullscreenRef.current = fullscreen;
      const fromRect = fullscreen ? fullscreenStartRectRef.current : fullscreenRectRef.current;
      const toRect = player?.getBoundingClientRect();
      if (fullscreen && toRect) fullscreenRectRef.current = toRect;
      if (fromRect && toRect) animateFullscreenRect(player, fromRect, toRect);
      fullscreenStartRectRef.current = null;
      setIsFullscreen(fullscreen);
    };
    document.addEventListener("fullscreenchange", onChange);
    document.addEventListener("webkitfullscreenchange", onChange);
    return () => {
      document.removeEventListener("fullscreenchange", onChange);
      document.removeEventListener("webkitfullscreenchange", onChange);
    };
  }, []);

  // ポーリングコールバック内で現在閲覧中の動画idを最新状態で参照 (stale closure防止)
  useEffect(() => {
    selectedVideoIdRef.current = selectedVideo?.id ?? null;
  }, [selectedVideo]);

  // 現在の動画が分析中かどうか
  const isAnalyzing =
    !!selectedVideo && analyzingJobs.some((j) => j.videoId === selectedVideo.id);

  // 分析中の進捗更新タイマー
  useEffect(() => {
    if (analyzingJobs.length === 0) return;
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, [analyzingJobs.length]);

  // タスク別進捗率(%)とステージ文言。
  // バックエンドがデコード・選別・推論・レンダリングの全段階で「実測」進捗度と現在のステージを報告するため
  // 架空の時間ベース推定は行わない。残り時間はステージ変更時に急変し混乱を招くため
  // 非表示とし、代わりに現在の処理内容(ステージ)とスムーズに遷移する進捗率を表示。

  // 分析ステージ文言 — サーバー(model/inference.py _report, backend prediction_job)の stage 値。
  // main は「〜処理中」(後続にアニメーションドットが付加), extra は補足情報。
  const PHASE_TEXT = {
    loading: () => ({ main: "分析エンジン準備中" }),
    decode: () => ({ main: "分析エンジン準備中" }),
    motion: () => ({ main: "車両の動きを計算中" }),
    coarse: () => ({ main: "映像全体をスクリーニング中" }),
    refine: (n, i) => ({ main: n > 1 && i ? `疑い区間 ${i}/${n} を詳細分析中` : "疑い区間を詳細分析中" }),
    scan: () => ({ main: "映像分析中" }),
    events: () => ({ main: "事故区間を整理中" }),
    render: (n, i) => ({ main: n > 1 && i ? `事故クリップ ${i}/${n} を生成中` : "事故クリップを生成中" }),
  };

  // 「〜処理中」後続のドット: . → .. → ... → .... を0.5秒間隔で反復。
  // ドット領域は固定幅(.phase-dots)のためドット数の変動による後続文字の揺れを防止。
  const phaseText = (pct, main, extra) => (
    <>
      {pct != null && `${Math.round(pct)}% · `}
      {main}
      <span className="phase-dots">{".".repeat(1 + (Math.floor(now / 500) % 4))}</span>
      {extra && ` · ${extra}`}
    </>
  );

  const jobProgress = (job) => {
    const hasReal = typeof job.progress === "number" && (job.progress > 0 || !!job.phase);
    // ステージ: 要求(サーバーキューで待機中) → 準備 → 進行(サーバーがステージ・進捗度を送信)
    // 他ユーザーのタスク情報は開示しないため待機順位は非表示。
    const stage = hasReal ? "running" : job.status === "PENDING" ? "queued" : "starting";

    // 進捗率はサーバーからの受信値をそのまま表示。
    const serverP = hasReal ? Math.min(99, job.progress) : 0;
    const pct = serverP;

    const ph = PHASE_TEXT[job.phase]?.(job.phaseCount, job.phaseIndex) ?? { main: "分析中" };
    // 上部ヘッダー領域: ステージ文言(中央)と進捗率(最右端, pctText)を個別に表示
    const statusText =
      stage === "queued" ? phaseText(null, "サーバーに分析を要求中")
        : stage === "starting" ? phaseText(null, "分析準備中")
          : phaseText(null, ph.main, ph.extra);
    const pctText = stage === "running" ? `${Math.round(pct)}%` : null;
    // サーバーのキュー待機中であることを伝える目的のため、要求ステージの文言は省略しない
    const shortText =
      stage === "queued" ? "サーバーに分析をリクエスト中" : stage === "starting" ? "準備中" : `${Math.round(pct)}%`;
    // 展開リストの下段行 — 上段行(短縮表示: %)と重複しないようステージのみ
    const detailText =
      stage === "queued" ? "サーバーの先行分析完了後に自動開始されます"
        : stage === "starting" ? phaseText(null, "分析準備中") : phaseText(null, ph.main, ph.extra);
    return { pct, preparing: stage !== "running", stage, statusText, pctText, shortText, detailText };
  };

  // オーバーレイ div 内で実際の動画コンテンツが占有する領域(object-fit:contain レターボックス)と
  // 原本解像度→画面スケールを返却。bbox描画/クリック座標換算を本関数で統一。
  // 画面座標(ポインター位置)に存在する車両検知ボックス — なければ null
  const detectedBoxAt = (e) => {
    const geom = getVideoGeom();
    const el = bboxOverlayRef.current;
    if (!geom || !el || detectedBoxes.length === 0) return null;
    const rect = el.getBoundingClientRect();
    const realX = (e.clientX - rect.left - geom.offsetX) / geom.scale;
    const realY = (e.clientY - rect.top - geom.offsetY) / geom.scale;
    return detectedBoxes.find(({ bbox: [x1, y1, x2, y2] }) => realX >= x1 && realX <= x2 && realY >= y1 && realY <= y2) || null;
  };

  // 動画領域サイズ(w, h) → 原本解像度との倍率・レターボックス余白。解像度はサーバーが分析に使用するサイズ(DB width/height)
  const geomFor = (w, h) => {
    const vw = selectedVideo?.width || w;
    const vh = selectedVideo?.height || h;
    const scale = Math.min(w / vw, h / vh);
    return {
      scale,
      vw,
      vh,
      offsetX: (w - vw * scale) / 2,
      offsetY: (h - vh * scale) / 2,
    };
  };

  const getVideoGeom = () => {
    const el = bboxOverlayRef.current;
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    return geomFor(rect.width, rect.height);
  };

  const renderGeom = overlaySize ? geomFor(overlaySize.w, overlaySize.h) : null;

  // 選択された車両ボックス(bboxList)は元動画解像度座標で保存。
  // 動画領域サイズが変動しても(ウィンドウサイズ · 右パネル · シアターモード · 全画面)描画時に現在サイズへ変換するため
  // ボックスが動画の同一位置に維持され、分析へ送信する座標も一貫。
  // 画面(オーバーレイ)座標 → 原本座標 (動画外のレターボックスへはみ出た部分は動画内にクリッピング)
  const toOrigBox = (x1, y1, x2, y2) => {
    const g = getVideoGeom();
    if (!g) return null;
    const ox = (x) => Math.round(Math.min(g.vw, Math.max(0, (x - g.offsetX) / g.scale)));
    const oy = (y) => Math.round(Math.min(g.vh, Math.max(0, (y - g.offsetY) / g.scale)));
    return { xmin: ox(x1), ymin: oy(y1), xmax: ox(x2), ymax: oy(y2) };
  };

  const toDispBox = (box) => {
    const g = renderGeom;
    if (!g) return null;
    return {
      left: g.offsetX + box.xmin * g.scale,
      top: g.offsetY + box.ymin * g.scale,
      width: (box.xmax - box.xmin) * g.scale,
      height: (box.ymax - box.ymin) * g.scale,
    };
  };

  // 幅広画面の動画再生画面ではボタン群を左ペイン最上部(事故クリップサイドバー上)に配置。
  // 狭幅画面(1024px以下)は縦積みの1列となりサイドバーが動画下部に配置されるため動画上部ヘッダーに配置。
  const isWideScreen = useMediaQuery("(min-width: 1025px)");
  const isMobilePlayback = useMediaQuery("(max-width: 720px)");
  const isCompactScreen = useMediaQuery("(max-width: 480px)");
  const useCompactFullscreenSettings = isFullscreen && isCompactScreen;
  const actionsInSide = isWideScreen && !!selectedVideo;

  useEffect(() => {
    if (!useCompactFullscreenSettings) setIsPlayerSettingsOpen(false);
  }, [useCompactFullscreenSettings]);

  useEffect(() => {
    if (!isPlayerSettingsOpen) return undefined;
    const onPointerDown = (event) => {
      if (!playerSettingsRef.current?.contains(event.target)) setIsPlayerSettingsOpen(false);
    };
    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        setIsPlayerSettingsOpen(false);
      }
    };
    document.addEventListener("pointerdown", onPointerDown, true);
    document.addEventListener("keydown", onKeyDown, true);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown, true);
      document.removeEventListener("keydown", onKeyDown, true);
    };
  }, [isPlayerSettingsOpen]);

  // 進行ステップ(01 車両選択 → 02 事故区間検出 → 03 クリップ確認)。
  // 幅広画面(1280px以上)は動画右ペイン上部に垂直配置、それ未満では操作ヘッダー内に水平配置。
  const isXLScreen = useMediaQuery("(min-width: 1280px)");
  // 現在実行すべきステップを画面状態から判定: 分析中 → 02, 車両選択中 → 01, 結果あり → 03,
  // ボックス指定済み → 02(検出実行待ち), その他 → 01。完了ステップはチェック(✓)、後続ステップは薄暗く表示。
  const currentStep = !selectedVideo ? 1
    : isAnalyzing ? 2
      : selectionMode !== "none" ? 1
        : selectedVideo.events.length > 0 ? 3
          : bboxList.length > 0 ? 2
            : 1;
  const selectedVideoJob = selectedVideo ? analyzingJobs.find((j) => j.videoId === selectedVideo.id) : null;
  // 車両検知・手動指定の実行中 — 動画サイズ・アスペクト比が変動すると描画中のボックス位置がズレるため
  // この間は動画サイズを変更する機能(シアターモード・全画面・右パネル開閉)を無効化
  const isChoosingVehicle = selectionMode !== "none" || isDetectingVehicles;
  const camEnabled = clipView === "cam" && !isChoosingVehicle;
  const camShown = camEnabled && selectedVideo
    ? (isMobilePlayback
      ? clipEventsAhead(selectedVideo.events, currentTime, currentEventId, 2).slice(0, 2)
      : clipEventsAhead(selectedVideo.events, currentTime, currentEventId, 3))
    : [];
  const camKey = camShown.map((e) => e.id).join(",");

  // 元動画の再生時刻とCAMクリップを同期させ現在区間に該当するクリップのみ重畳表示。
  useEffect(() => {
    const main = videoElRef.current;
    const overlay = bboxOverlayRef.current;
    if (!main || !overlay || !camKey) return undefined;
    const events = selectedVideo?.events || [];
    const layers = [...overlay.querySelectorAll(".cam-clip-layer")]
      .map((layer) => ({
        layer,
        cam: layer.querySelector("video"),
        ev: events.find((e) => e.id === Number(layer.dataset.eventId)),
      }))
      .filter((x) => x.cam && x.ev);
    const chips = [camChipRef.current, camOverlayChipRef.current].filter(Boolean);
    // CAMはアクティブ区間に進入した際に1度のみ同期。seek中も元動画は再生され続けるため毎フレーム
    // 現在時刻で再アラインするとモバイル環境でCAM seekが無限ループし再生が開始されなくなるのを防止。
    const blockedCamPlayback = new WeakSet();
    const alignedCam = new WeakSet();
    const aligningCam = new WeakSet();
    const sync = () => {
      const now = clipEventAt(events, main.currentTime, currentEventId);
      let shown = null;
      layers.forEach(({ layer, cam, ev }) => {
        const local = main.currentTime - ev.clipStartSec;
        const inClip = local >= 0 && local < (cam.duration || 0);
        const activeEvent = now?.id === ev.id;
        const eventStart = ev.startSec ?? ev.timestamp;
        const prepareAhead = isMobilePlayback && !activeEvent
          && eventStart > main.currentTime && eventStart - main.currentTime <= 2;
        // モバイルではイベント進入前にCAMを目的フレームへseekして一時停止。
        // 実際の進入時の初期seek遅延を回避しつつ、プレビュー再生によるデコーダー負荷増加を防止。
        const shouldSyncClip = inClip && (activeEvent || !isMobilePlayback);
        if (prepareAhead) {
          if (cam.seeking) {
            layer.classList.remove("shown");
            return;
          }
          const targetLocal = Math.max(0, eventStart - ev.clipStartSec);
          if (cam.duration > 0 && targetLocal < cam.duration && !alignedCam.has(cam)) {
            if (!aligningCam.has(cam) && Math.abs(cam.currentTime - targetLocal) > 0.08) {
              if (!cam.paused) cam.pause();
              aligningCam.add(cam);
              cam.currentTime = targetLocal;
              layer.classList.remove("shown");
              return;
            }
            if (aligningCam.has(cam)) {
              // seeked イベントで到達したイベント開始フレームを保持。
              aligningCam.delete(cam);
              alignedCam.add(cam);
            } else {
              alignedCam.add(cam);
            }
          }
          if (!cam.paused) cam.pause();
        } else if (shouldSyncClip) {
          if (cam.seeking) {
            layer.classList.remove("shown");
            return;
          }
          if (aligningCam.has(cam)) {
            aligningCam.delete(cam);
            alignedCam.add(cam);
          } else if (!alignedCam.has(cam)) {
            if (Math.abs(cam.currentTime - local) > 0.08) {
              if (!cam.paused) cam.pause();
              aligningCam.add(cam);
              cam.currentTime = local;
              layer.classList.remove("shown");
              return;
            }
            alignedCam.add(cam);
          }
          if (cam.playbackRate !== main.playbackRate) cam.playbackRate = main.playbackRate;
          if (!main.paused && cam.paused && !blockedCamPlayback.has(cam)) {
            cam.play().catch(() => blockedCamPlayback.add(cam));
          }
          if (main.paused && !cam.paused) cam.pause();
        } else if (!cam.paused) {
          cam.pause();
        }
        if (!shouldSyncClip && !prepareAhead && (!inClip || isMobilePlayback)) {
          alignedCam.delete(cam);
          aligningCam.delete(cam);
        }
        // CAM動画がモバイルポリシー等で再生されない場合、停止フレームを被せず元動画を表示。
        const show = activeEvent && inClip && cam.readyState >= 2
          && !cam.seeking && alignedCam.has(cam)
          && (main.paused || !cam.paused);
        layer.classList.toggle("shown", show);
        if (show) shown = ev;
      });
      chips.forEach((chip) => {
        chip.hidden = !shown;
        if (shown) {
          const label = shown.analysisNo != null ? `分析 #${shown.analysisNo} 衝突表示` : "衝突表示";
          if (chip.textContent !== label) chip.textContent = label;
        }
      });
    };
    let raf = 0;
    const loop = () => { sync(); raf = requestAnimationFrame(loop); };
    const onPlay = () => {
      cancelAnimationFrame(raf);
      layers.forEach(({ cam }) => blockedCamPlayback.delete(cam));
      sync();
      raf = requestAnimationFrame(loop);
    };
    const onPause = () => { cancelAnimationFrame(raf); sync(); };
    const onMainSeeking = () => {
      layers.forEach(({ cam }) => {
        alignedCam.delete(cam);
        aligningCam.delete(cam);
        if (!cam.paused) cam.pause();
      });
      sync();
    };
    const mainEvents = { play: onPlay, pause: onPause, seeking: onMainSeeking, seeked: sync, ratechange: sync };
    const camEvents = { loadedmetadata: sync, loadeddata: sync, seeked: sync };
    Object.entries(mainEvents).forEach(([k, fn]) => main.addEventListener(k, fn));
    layers.forEach(({ cam }) => Object.entries(camEvents).forEach(([k, fn]) => cam.addEventListener(k, fn)));
    if (main.paused) sync(); else onPlay();
    return () => {
      cancelAnimationFrame(raf);
      Object.entries(mainEvents).forEach(([k, fn]) => main.removeEventListener(k, fn));
      layers.forEach(({ cam }) => Object.entries(camEvents).forEach(([k, fn]) => cam.removeEventListener(k, fn)));
      chips.forEach((chip) => { chip.hidden = true; });
    };
  }, [camKey, selectedVideo, currentEventId, isMobilePlayback]);

  // 分析履歴から選択した分析のクリップが残存している場合は「強調表示中」 (全クリップ削除時に自動解除)
  const isHighlighting = !!selectedVideo && highlightAnalysisNo != null
    && selectedVideo.events.some((e) => e.analysisNo === highlightAnalysisNo);
  // 左ペインリストは最新分析を優先、同一分析内では動画時刻順にソート表示。
  const listedEvents = selectedVideo
    ? [...selectedVideo.events].sort((a, b) => (b.analysisNo ?? 0) - (a.analysisNo ?? 0) || a.startSec - b.startSec)
    : [];
  const workflowSteps = selectedVideo ? (
    <ol className="incident-workflow" aria-label="事故疑い区間の確認フロー">
      {[
        { n: 1, title: "分析対象の車両選択", desc: "自動検出または手動指定" },
        {
          n: 2,
          title: "事故区間の検出",
          desc: isAnalyzing
            ? `分析中 · ${typeof selectedVideoJob?.progress === "number" ? `${Math.round(selectedVideoJob.progress)}%` : "準備中"}`
            : "選択車両の接触シーンを探索",
        },
        { n: 3, title: "クリップ確認", desc: "原本と疑いシーンを併せて確認", count: selectedVideo.events.length },
      ].map((st) => {
        const state = st.n < currentStep ? "is-complete" : st.n === currentStep ? "is-current" : "is-pending";
        return (
          <li key={st.n} className={state} aria-current={st.n === currentStep ? "step" : undefined}>
            <span className="step-num" aria-hidden="true">
              {st.n < currentStep ? (
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5" /></svg>
              ) : String(st.n).padStart(2, "0")}
            </span>
            <div><strong>{st.title}</strong><small>{st.desc}</small></div>
            {st.count > 0 && <b>{st.count}件</b>}
          </li>
        );
      })}
    </ol>
  ) : null;

  // 動画再生画面の 車両検知 · 車両手動指定 · 事故検知実行 ボタン群。
  // 車両検知ボタンは状態に応じて3段階に変化:
  // 検知前「車両検知」 → 検知中「検知キャンセル」 → 検知後の車両選択中「車両選択終了」
  const watchActionButtons = selectedVideo ? (
    <div className="watch-header-actions">
      <button
        className={`auto-detect-btn ${selectionMode === "auto" ? "active" : ""} ${isDetectingVehicles ? "detecting" : ""}`}
        disabled={isAnalyzing}
        onClick={isDetectingVehicles ? handleCancelDetectVehicles
          : selectionMode === "auto" ? handleEndDesignateMode : handleAutoDetectVehicles}
        title={isDetectingVehicles ? "進行中の車両検出をキャンセルします"
          : selectionMode === "auto" ? "車両選択を終了します"
            : "現在画面の車両を自動検出して表示します"}
      >
        {isDetectingVehicles ? "🛑 検出キャンセル" : selectionMode === "auto" ? "車両選択終了" : "車両検出"}
      </button>

      {/* 車両手動指定 + 指定モード案内ツールチップ(ボタン上部にフロート表示し上部バーの空白にオーバーレイ — 他のレイアウトを干渉しない) */}
      <div className="designate-btn-wrapper">
        <button
          className={`bbox-designate-btn ${selectionMode === "manual" ? "active" : ""}`}
          disabled={isAnalyzing || isDetectingVehicles}
          onClick={() => {
            if (selectionMode === "manual") {
              // 「手動指定終了」はモードのみ終了 — 描画済みボックスは保持(車両検知で選択したボックスと同等)。
              // 削除する場合はボックスにホバーして ✕
              handleEndDesignateMode();
            } else {
              handleStartManualDesignate();
            }
          }}
          title={isAnalyzing ? "分析中は指定できません"
            : isDetectingVehicles ? "車両検出中は指定できません"
              : (selectionMode === "manual" ? "手動指定モードを終了します" : "マウスのドラッグで車両を手動指定します")}
        >
          {selectionMode === "manual" ? "手動指定終了" : "車両を手動指定"}
        </button>
        {selectionMode === "manual" && (
          <div className="designate-hint" role="status">
            🖱️ マウスでドラッグして事故対象車両を指定してください。
          </div>
        )}
      </div>


      {/* 事故検知実行ボタン + 確認ポップオーバー / キャンセルボタン */}
      <div className="detect-btn-wrapper">
        {isAnalyzing ? (
          <button
            className="detect-run-btn analyzing-cancel-btn"
            onClick={() => {
              const currentJob = analyzingJobs.find((j) => j.videoId === selectedVideo?.id);
              analyzeAbortControllerRef.current?.abort();
              analyzeAbortControllerRef.current = null;
              handleCancelAnalysis(currentJob?.taskId);
            }}
            title="進行中の事故検知分析をキャンセルします"
          >
            🛑 事故検知キャンセル
          </button>
        ) : (
          <button
            className="detect-run-btn"
            onClick={handleAnalyzeClick}
            title="選択した車両の事故疑い区間を検出します"
          >
            事故区間を検出
          </button>
        )}

        {showAnalyzeConfirm && (
          <>
            <div
              className="analyze-popover-overlay"
              onClick={() => setShowAnalyzeConfirm(false)}
            />
            <div className="analyze-popover">
              <p className="analyze-popover-text">
                {selectedVideo?.analyzedBefore
                  ? "すでに分析済みの動画です。再分析を実行しますか？"
                  : "選択した車両の接触疑い区間を検出しますか？"}
              </p>
              <div className="analyze-popover-actions">
                <button className="analyze-popover-run" onClick={runAnalysis}>
                  実行
                </button>
                <button
                  className="analyze-popover-cancel"
                  onClick={() => setShowAnalyzeConfirm(false)}
                >
                  戻る
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  ) : null;

  return (
    <div
      className={`dashboard-layout ${selectionMode === "manual" && isDrawing ? "bbox-drawing" : ""}`}
      onDragStartCapture={(e) => {
        if (selectionMode === "manual" && isDrawing) e.preventDefault();
      }}
    >
      {/* 上部ナビゲーションバー */}
      <header className="top-navbar">
        <div className="nav-section nav-left">
          <button className="nav-logo nav-home-btn" onClick={handleBackToHome} aria-label="SIOTホームへ移動">
            <span className="workspace-wordmark" translate="no">SI<span className="workspace-lens">O</span>T<span className="workspace-brand-dot">.</span></span><span className="workspace-brand-caption" translate="no">{selectedVideo ? "INCIDENT REVIEW" : "LIBRARY"}</span>
          </button>
        </div>

        <div className="nav-section nav-center">
          {analyzingJobs.length === 0 ? (
            <div className="analysis-status-box analysis-status-idle">
              <span className="analysis-idle-text">現在、分析中の動画はありません</span>
            </div>
          ) : (() => {
            const lead = analyzingJobs[0];
            const { pct, preparing, statusText, pctText } = jobProgress(lead);
            return (
              <div className="analysis-status-box">
                <div className="analysis-status-main">
                  <div className="analysis-status-info">
                    <span className="analysis-status-label">
                      {/* 動画リストのタイトルと統一 (タイトル設定時はそのタイトル、未設定時は「録画日 録画データ #N」) */}
                      🔍 {(() => {
                        const v = videos.find((x) => x.id === lead.videoId);
                        return v ? getVideoDisplayTitle(v, videos) : lead.dateLabel;
                      })()} 分析中
                    </span>
                    <span className="analysis-status-eta">
                      {statusText}
                    </span>
                    {pctText && <span className="analysis-status-pct">{pctText}</span>}
                  </div>
                  <div className="analysis-progress-track">
                    <div
                      className={`analysis-progress-fill${preparing ? " is-indeterminate" : ""}`}
                      style={preparing ? undefined : { width: `${pct}%` }}
                    />
                  </div>
                </div>

                <button
                  className="analysis-jobs-toggle"
                  onClick={() => setShowJobsDropdown((v) => !v)}
                  title="分析中の動画一覧を表示します"
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
                        分析中の動画 ({analyzingJobs.length})
                      </div>
                      {analyzingJobs.map((job) => {
                        const p = jobProgress(job);
                        return (
                          <div className="analysis-job-row" key={job.taskId ?? job.startedAt}>
                            <div className="analysis-job-top">
                              <span className="analysis-job-name">
                                {(() => {
                                   // 同一日付の動画が複数の場合は「録画データ #N」で識別表示
                                  const v = videos.find((x) => x.id === job.videoId);
                                  return v ? getVideoDisplayTitle(v, videos) : job.dateLabel;
                                })()}
                              </span>
                              <div className="analysis-job-top-right">
                                <span className="analysis-job-pct">
                                  {p.shortText}
                                </span>
                                <button
                                  className="analysis-job-cancel-btn"
                                  onClick={() => {
                                    analyzeAbortControllerRef.current?.abort();
                                    analyzeAbortControllerRef.current = null;
                                    handleCancelAnalysis(job.taskId);
                                  }}
                                  title="この分析をキャンセルします"
                                >
                                  🛑 キャンセル
                                </button>
                              </div>
                            </div>
                            <div className="analysis-progress-track">
                              <div
                                className={`analysis-progress-fill${p.preparing ? " is-indeterminate" : ""}`}
                                style={p.preparing ? undefined : { width: `${p.pct}%` }}
                              />
                            </div>
                            <span className="analysis-job-eta">
                              {p.detailText}
                            </span>
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
              aria-expanded={isProfileOpen}
              aria-controls="profile-menu"
            >
              <div className="profile-avatar">
                <span>{avatarInitial}</span>
              </div>
              <span className="profile-trigger-copy"><strong>{displayName}</strong><small>マイアカウント</small></span><span className="profile-trigger-chevron" aria-hidden="true">⌄</span>
            </button>

            {isProfileOpen && (
              <>
                <div className="dropdown-overlay" onClick={() => setIsProfileOpen(false)} />
                <div className="profile-dropdown-menu" id="profile-menu">
                  {/* 管理者/ユーザー情報要約 Header */}
                  <div className="dropdown-header">
                    <div className="header-avatar">{avatarInitial}</div>
                    <div className="header-info">
                      <span className="info-name">{displayName} ({currentUsername})</span>
                      <span className="info-role">{roleLabel}</span>
                    </div>
                  </div>

                  <div className="dropdown-divider" />

                  {/* アカウント設定 (Account Settings) */}
                  <div className="dropdown-section-title">アカウント設定</div>
                  <button
                    className="dropdown-item"
                    onClick={() => {
                      setIsProfileOpen(false);
                      openSettings();
                      setActiveSettingsTab("account");
                    }}
                  >
                    <InterfaceIcon name="user" /> {currentUser?.role === "ADMIN" ? "管理者情報の変更" : "ユーザー情報の変更"}
                  </button>
                  <button
                    className="dropdown-item"
                    onClick={() => {
                      setIsProfileOpen(false);
                      openSettings();
                      setActiveSettingsTab("security");
                    }}
                  >
                    <InterfaceIcon name="lock" /> パスワード変更
                  </button>

                  <div className="dropdown-divider" />

                  {/* テーマ設定 (Appearance) */}
                  <div className="dropdown-section-title">テーマ設定</div>
                  <div className="dropdown-item-toggle">
                    <span className="theme-label"><InterfaceIcon name="moon" /> ダークモード</span>
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

                  {/* ログアウト (Logout) */}
                  <button
                    className="dropdown-item logout-item"
                    onClick={() => {
                      setIsProfileOpen(false);
                      onLogout();
                    }}
                  >
                    <InterfaceIcon name="exit" /> ログアウト
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      </header>

      {/* コンテンツ領域: 動画一覧または再生画面 */}
      {!selectedVideo ? (
        <main className="home-view">
          <section className="incident-overview">
            <div className="incident-overview-copy"><span className="incident-eyebrow" translate="no">SIOT / INCIDENT REVIEW</span><h1>駐車の瞬間から、<br />見逃さない事故の手がかり。</h1><p>防犯カメラ映像から接触の疑いがある場面を検出し、必要な区間を素早く確認できます。</p><button className="incident-upload" onClick={() => setShowUpload(true)}>動画を追加して分析開始 <span aria-hidden="true">↗</span></button></div>
            <div className="incident-metrics" aria-label="動画ステータス">
              <div><span>保存動画</span><strong>{videos.length}<small>件</small></strong></div>
              <div><span>事故疑い区間</span><strong>{videos.reduce((sum, video) => sum + video.events.length, 0)}<small>件</small></strong></div>
              <div><span>分析進行中</span><strong>{analyzingJobs.length}<small>件</small></strong></div>
            </div>
          </section>
          <div className="incident-section-title"><div><span className="incident-eyebrow" translate="no">VIDEO LIBRARY</span><h2>動画保管庫</h2></div></div>
          {/* 期間フィルターおよび動画管理ツールバー */}
          <div className="home-toolbar">
            <div className="home-toolbar-filters">
              <div className="filter-pills">
                <button className={filterDays === 9999 ? "active" : ""} onClick={() => { setCurrentPage(1); setFilterDays(9999); }}>すべて</button>
                <button className={filterDays === 7 ? "active" : ""} onClick={() => { setCurrentPage(1); setFilterDays(7); }}>1週間</button>
                <button className={filterDays === 14 ? "active" : ""} onClick={() => { setCurrentPage(1); setFilterDays(14); }}>2週間</button>
                <button className={filterDays === 30 ? "active" : ""} onClick={() => { setCurrentPage(1); setFilterDays(30); }}>1ヶ月</button>
                <button className={filterDays === 90 ? "active" : ""} onClick={() => { setCurrentPage(1); setFilterDays(90); }}>3ヶ月</button>
              </div>

              <div className="list-search-bar">
                <span className="search-icon"><SearchIcon /></span>
                <input
                  type="text"
                  placeholder="動画の日付、タイトルを検索"
                  value={searchQuery}
                  onChange={(e) => { setCurrentPage(1); setSearchQuery(e.target.value); }}
                  aria-label="動画検索"
                />
                {searchQuery && (
                  <button className="search-clear-btn" onClick={() => { setCurrentPage(1); setSearchQuery(""); }} aria-label="検索ワードをクリア">
                    ✕
                  </button>
                )}
              </div>
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
                    🗑 選択削除 ({selectedForDelete.length})
                  </button>
                </>
              ) : (
                <>
                  <button className="toolbar-upload-btn" onClick={() => setShowUpload(true)}>
                    ⬆ 動画アップロード
                  </button>
                  <button className="toolbar-delete-btn" onClick={() => setDeleteMode(true)}>
                    🗑 動画削除
                  </button>
                </>
              )}
            </div>
          </div>


          {/* 動画サムネイルグリッド — カラム数はCSSが表示幅に合わせて決定し、1ページあたりの件数がそれに従う */}
          <div className="video-grid" ref={setVideoGridEl}>
            {filteredVideos.length === 0 ? (
              <div className="empty-state">
                {videoLoadError ? (
                  <>
                    <p>{videoLoadError}</p>
                    <button type="button" onClick={loadVideos} disabled={isLoadingVideos}>
                      {isLoadingVideos ? "動画一覧を読み込み中…" : "再試行"}
                    </button>
                  </>
                ) : (
                  "選択された期間に該当する動画がありません。"
                )}
              </div>
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
                      {/* 動画の先頭フレームサムネイル */}
                      <img
                        className="video-thumbnail-img"
                        src={api.thumbnailUrl(video.id)}
                        alt={`${video.date} サムネイル`}
                        loading="lazy"
                        onError={(e) => { e.currentTarget.style.display = "none"; }}
                      />
                      {/* 左下: 録画日 · 右下: 疑い区間数 */}
                      {video.date && <span className="recording-date-badge">{video.date}</span>}
                      <span className="event-count-badge">疑い区間 {video.events.length}件</span>
                      {deleteMode && (
                        <span className={`delete-check ${isSelected ? "checked" : ""}`}>
                          {isSelected ? "✓" : ""}
                        </span>
                      )}
                    </div>
                    <div className="video-info">
                      <div className="video-info-text">
                        <h3>{getVideoDisplayTitle(video, videos)}</h3>
                        <p>動画の長さ: {formatTime(video.duration)}</p>
                        {/* 保存期間 — 経過すると動画と分析クリップが削除される。7日以下は強調 */}
                        {video.expiresInDays != null && (
                          <p
                            className={`video-expiry ${video.expiresInDays <= 7 ? "soon" : ""}`}
                            title="保存期間を過ぎると動画と分析されたクリップが削除されます"
                          >
                            {video.expiresInDays > 0 ? `${video.expiresInDays}日後に期限切れ` : "まもなく期限切れ"}
                          </p>
                        )}
                      </div>
                      {/* ··· メニュー: 動画情報確認(情報ポップアップ — アップロード日時等を含む) / 動画削除 */}
                      {!deleteMode && <VideoCardActions video={video} title={getVideoDisplayTitle(video, videos)} onShowInfo={() => setInfoVideo(video)} onDeleted={(id) => {
                        setVideos((previous) => previous.filter((video) => video.id !== id));
                        setSelectedForDelete((previous) => previous.filter((videoId) => videoId !== id));
                        showToast("動画を削除しました。", "success");
                      }} />}
                    </div>
                  </div>
                );
              })
            )}
          </div>

          {/* ページネーション (動画数が6件を超える場合に表示) */}
          {filteredVideos.length > VIDEOS_PER_PAGE && (
            <div className="pagination-bar">
              {/* ボタン群は中央、ページ情報は右側 — 情報テキストによるボタン位置ズレを防ぐ */}
              <div className="pagination-controls">
                <button
                  className="pagination-btn pagination-nav"
                  onClick={() => handlePageChange(1)}
                  disabled={visiblePage === 1}
                  title="最初のページへ移動"
                >
                  «
                </button>
                <button
                  className="pagination-btn pagination-nav"
                  onClick={() => handlePageChange(Math.max(1, visiblePage - 1))}
                  disabled={visiblePage === 1}
                  title="前のページへ移動"
                >
                  ‹
                </button>

                <div className="pagination-pages">
                  {getPageNumbers().map((pageNum) => (
                    <button
                      key={pageNum}
                      className={`pagination-btn pagination-num ${visiblePage === pageNum ? "active" : ""}`}
                      onClick={() => handlePageChange(pageNum)}
                    >
                      {pageNum}
                    </button>
                  ))}
                </div>

                <button
                  className="pagination-btn pagination-nav"
                  onClick={() => handlePageChange(Math.min(totalPages, visiblePage + 1))}
                  disabled={visiblePage === totalPages}
                  title="次のページへ移動"
                >
                  ›
                </button>
                <button
                  className="pagination-btn pagination-nav"
                  onClick={() => handlePageChange(totalPages)}
                  disabled={visiblePage === totalPages}
                  title="最後のページへ移動"
                >
                  »
                </button>
              </div>

              <span className="pagination-info">
                {visiblePage} / {totalPages} ページ (総 {filteredVideos.length}件)
              </span>
            </div>
          )}
        </main>
      ) : (
        <main className={`watch-view ${isTheaterMode ? "theater-view" : ""}`}>
          {!isTheaterMode && <section className="incident-workspace-header">
            <div className="incident-workspace-top"><div><button className="incident-back" onClick={handleBackToHome}>← 動画保管庫</button><h1>{getVideoDisplayTitle(selectedVideo, videos)}</h1></div>
              {/* 右側パネル(進捗ステップ・分析履歴)の開閉 — 閉じるとその領域が消えて動画が広がる(幅広画面のみ)。
                  動画サイズが変わるため車両選択中は無効化 */}
              {isXLScreen && (
                <button
                  className="panel-toggle-btn"
                  onClick={() => setIsHistoryOpen((open) => !open)}
                  disabled={isChoosingVehicle}
                  aria-expanded={isHistoryOpen}
                  title={isChoosingVehicle ? "車両の選択中は利用できません"
                    : isHistoryOpen ? "進捗ステップ・分析履歴パネルを閉じて動画を広く表示します" : "進捗ステップ・分析履歴パネルを開きます"}
                >
                  {/* 右側パネル形状アイコン — 開いていれば右側が塗られ、閉じていれば空枠 */}
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" aria-hidden="true">
                    <rect x="3" y="4.5" width="18" height="15" rx="1.5" />
                    <path d="M15 4.5v15" />
                    {isHistoryOpen && <rect x="15" y="4.5" width="6" height="15" fill="currentColor" stroke="none" opacity="0.45" />}
                  </svg>
                  {isHistoryOpen ? "右側パネルを閉じる" : "右側パネルを開く"}
                </button>
              )}
            </div>
            {!isXLScreen && workflowSteps}
          </section>}

          {/* 狭い画面(1024px以下): 動画上のヘッダーにボタン群。
              広い画面はヘッダー行なしで動画を最上部に — ボタンは左側エリアの最上部に配置 */}
          {!actionsInSide && (
            <div className="watch-header">
              <div className="watch-header-right">
                {watchActionButtons}
              </div>
            </div>
          )}

          <div className={`watch-layout ${isTheaterMode ? "theater-mode" : ""} ${!isHistoryOpen && isXLScreen ? "history-collapsed" : ""}`}>
            {/* 左側エリア: (幅広画面) ボタン群 + 事故クリップサイドバー */}
            <div className="watch-side">
              {actionsInSide && watchActionButtons}
              <aside className={`event-sidebar ${isHighlighting ? "highlighting" : ""}`}>
                {/* ヘッダー: 上段タイトル + ボタン、下段件数案内 */}
                <div className="event-sidebar-header">
                  <h3>事故疑い区間</h3>
                  {/* 一括削除 */}
                  <div className="event-header-actions">
                    {selectedVideo.events.length > 0 && (
                      <button
                        className="event-clear-all-btn"
                        onClick={requestClearAllEvents}
                        title="この動画の事故疑いクリップをすべて削除します"
                      >
                        🗑 すべて削除
                      </button>
                    )}
                  </div>
                  <p className="event-summary">クリップ {selectedVideo.events.length}件 · 直接の確認が必要です。</p>
                </div>

                <div className="event-list" ref={eventListRef}>
                  {selectedVideo.events.length === 0 ? (
                    <div className="event-empty-hint">検出されたイベントはありません。</div>
                  ) : (
                    listedEvents.map((event) => (
                      <div
                        key={event.id}
                        className={`event-item ${currentEventId === event.id ? "active" : ""} ${
                          !isHighlighting
                            ? ""
                            : event.analysisNo === highlightAnalysisNo ? "analysis-highlight" : "analysis-dim"
                        }`}
                        onClick={() => seekToEvent(event)}
                      >
                        <div className="event-item-top">
                          <span className="event-time-pill">{formatTime(event.timestamp)}</span>
                          <span className="event-actual-time-text" title="この動画の何回目の分析で生成されたクリップかを示します">
                            {event.analysisNo != null && `分析 #${event.analysisNo}`}
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
                                  showToast("このイベントの事故クリップはまだありません。", "warning");
                                }
                              }}
                              aria-label={`${formatTime(event.timestamp)} 事故区間クリップ再生`}
                              title="事故区間のクリップを表示します"
                            >
                              {/* 文字 ▶ はフォント依存や絵文字化を防ぐためSVGで描画 */}
                              <PlayerIcon name="play" size={14} />
                            </button>
                            <button
                              className="event-delete-item-btn"
                              onClick={(e) => requestDeleteEvent(event, e)}
                              title="このクリップを削除します"
                              aria-label={`${formatTime(event.timestamp)} 事故区間クリップ削除`}
                            >
                              <TrashIcon size={15} />
                            </button>
                          </div>
                        </div>
                        <div className="event-item-bottom">
                          <h4 className="event-title-text" title="分析日時">
                            {(() => {
                              const at = formatAnalyzedAt(event.analyzedAt);
                              return at && (
                                <>
                                  {at[0]}
                                  <span className="event-analyzed-time">{at[1]}</span>
                                </>
                              );
                            })()}
                          </h4>
                        </div>
                      </div>
                    ))
                  )}
                </div>


                {/* カレンダーをサイドバーイベントリスト下部に挿入 */}
                <div className="sidebar-calendar-container">
                  <section className="event-calendar-panel" style={{ background: "transparent", padding: 0 }}>
                    <div className="calendar-header">
                      <button className="calendar-nav-btn" onClick={() => moveCalendarMonth(-1)} aria-label="前月">
                        ‹
                      </button>
                      <h2>{formatMonthLabel(calendarMonth)}</h2>
                      <button className="calendar-nav-btn" onClick={() => moveCalendarMonth(1)} aria-label="次月">
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

                        // 動画の総イベント件数を合算して強度を計算
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
                            title={hasVideo ? `合計 ${videos.length}件の動画 / ${totalEvents}件のイベント` : "動画なし"}
                          >
                            <span className="calendar-date-number">{date.getDate()}</span>
                            {videos.length > 1 && (
                              <span className="calendar-multi-count-badge" title={`動画 ${videos.length}件`}>
                                {videos.length}件
                              </span>
                            )}
                          </button>
                        );
                      })}
                    </div>
                  </section>
                  <div className="calendar-legend"><span><i /> 動画あり</span><span><i /> 疑い区間あり</span><span><i /> 選択した日付</span></div>
                </div>
              </aside>
            </div>

            {/* 右側/中央: メイン動画プレイヤー */}
            <section className="player-section">
              {isTheaterMode && (
                <div className="theater-mobile-heading">
                  <div className="theater-mobile-heading-copy">
                    <span>シアターモード</span>
                    <strong>{getVideoDisplayTitle(selectedVideo, videos)}</strong>
                  </div>
                  <button type="button" className="theater-mobile-exit" aria-label="シアターモードを終了" onClick={() => setIsTheaterMode(false)}>
                    閉じる <span aria-hidden="true">×</span>
                  </button>
                </div>
              )}
              {/* 前回の分析が失敗・中断した場合は同一車両ボックスで再実行を案内 */}
              {!isAnalyzing && selectedVideo?.lastTask?.status === "FAILURE" && (
                <div
                  className="analysis-failed-banner"
                  role="status"
                  title={selectedVideo.lastTask.errorMessage || ""}
                >
                  <span>
                    {(selectedVideo.lastTask.errorMessage || "").startsWith("分析サーバーが再起動")
                      ? "⚠️ サーバー再起動により分析が中断されました"
                      : "⚠️ 前回の分析に失敗しました"}
                  </span>
                  <button className="analysis-retry-btn" onClick={handleRetryAnalysis}>
                    再試行
                  </button>
                </div>
              )}

              <div className={`player-container ${isViewportFullscreen ? "is-viewport-fullscreen" : ""} ${isFullscreen ? "is-fullscreen-active" : ""}`} ref={playerContainerRef}>
                <span className="cam-overlay-chip" ref={camOverlayChipRef} role="status" hidden />
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
                  onClick={(e) => {
                    if (isBBoxMode || isDetectingVehicles || e.target.closest("button")) return;
                    const gesture = videoPointerGestureRef.current;
                    videoPointerGestureRef.current = null;
                    if (gesture?.moved) return;
                    if (usesTouchVideoControls()) {
                      toggleVideoTapControls();
                    } else {
                      toggleDesktopVideoPlayback();
                    }
                  }}
                  /* ボックス描画・車両選択は pointer イベント — マウス・タッチ・ペン共通で動作 */
                  onPointerDown={(e) => {
                    if (!isBBoxMode) {
                      if (e.isPrimary && e.button === 0 && !e.target.closest("button")) {
                        videoPointerGestureRef.current = { id: e.pointerId, x: e.clientX, y: e.clientY, moved: false };
                      }
                      return;
                    }
                    if (e.target.closest("button")) return; // ボックスの✕等のボタン操作時は描画を開始しない
                    // タッチ操作時はタップ位置のボックスを直接検出
                    const picked = selectionMode === "auto" ? hoveredDetectedBox || detectedBoxAt(e) : null;
                    if (selectionMode === "auto" && picked) {
                      const [x1, y1, x2, y2] = picked.bbox; // 検出結果はすでに元解像度の座標
                      setBboxList([{ id: ++bboxSeq, xmin: x1, ymin: y1, xmax: x2, ymax: y2 }]);
                      showToast("車両が選択されました。", "success");
                      return;
                    }
                    if (selectionMode === "manual") {
                      e.preventDefault();
                      window.getSelection()?.removeAllRanges();
                      // ポインターが画面外に出ても追尾するようキャプチャ
                      e.currentTarget.setPointerCapture?.(e.pointerId);
                      const rect = bboxOverlayRef.current.getBoundingClientRect();
                      const x = Math.round(e.clientX - rect.left);
                      const y = Math.round(e.clientY - rect.top);
                      setIsDrawing(true);
                      setCurrentDraw({ startX: x, startY: y, endX: x, endY: y });
                    }
                  }}
                  onPointerMove={(e) => {
                    if (!isBBoxMode) {
                      const gesture = videoPointerGestureRef.current;
                      if (gesture?.id === e.pointerId && Math.hypot(e.clientX - gesture.x, e.clientY - gesture.y) > 5) {
                        gesture.moved = true;
                      }
                      return;
                    }
                    const rect = bboxOverlayRef.current.getBoundingClientRect();

                    if (selectionMode === "manual" && isDrawing) {
                      const x = Math.round(e.clientX - rect.left);
                      const y = Math.round(e.clientY - rect.top);
                      setCurrentDraw((prev) => (prev ? { ...prev, endX: x, endY: y } : null));
                      return;
                    }

                    if (selectionMode === "auto" && detectedBoxes.length > 0 && getVideoGeom()) {
                      const { scale, offsetX, offsetY } = getVideoGeom();
                      // 画面(オーバーレイ)座標 → レターボックス余白除去 → 元解像度座標
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
                  onPointerUp={(e) => {
                    if (e.currentTarget.hasPointerCapture?.(e.pointerId)) {
                      e.currentTarget.releasePointerCapture(e.pointerId);
                    }
                    if (selectionMode === "manual" && isDrawing && currentDraw) {
                      const rect = bboxOverlayRef.current.getBoundingClientRect();
                      const x = Math.round(e.clientX - rect.left);
                      const y = Math.round(e.clientY - rect.top);
                      const xmin = Math.min(currentDraw.startX, x);
                      const ymin = Math.min(currentDraw.startY, y);
                      const xmax = Math.max(currentDraw.startX, x);
                      const ymax = Math.max(currentDraw.startY, y);
                      const orig = xmax - xmin > 5 && ymax - ymin > 5 ? toOrigBox(xmin, ymin, xmax, ymax) : null;
                      if (orig) setBboxList([{ id: ++bboxSeq, ...orig }]);
                      setCurrentDraw(null);
                      setIsDrawing(false);
                    }
                  }}
                  onPointerCancel={() => {
                    videoPointerGestureRef.current = null;
                    // システムジェスチャー等で途切れた場合は描画中のボックスを破棄
                    setCurrentDraw(null);
                    setIsDrawing(false);
                  }}
                  onPointerLeave={() => {
                    if (selectionMode === "manual" && isDrawing && currentDraw) {
                      const xmin = Math.min(currentDraw.startX, currentDraw.endX);
                      const ymin = Math.min(currentDraw.startY, currentDraw.endY);
                      const xmax = Math.max(currentDraw.startX, currentDraw.endX);
                      const ymax = Math.max(currentDraw.startY, currentDraw.endY);
                      const orig = xmax - xmin > 5 && ymax - ymin > 5 ? toOrigBox(xmin, ymin, xmax, ymax) : null;
                      if (orig) setBboxList([{ id: ++bboxSeq, ...orig }]);
                    }
                    setCurrentDraw(null);
                    setIsDrawing(false);
                    setHoveredDetectedBox(null);
                  }}
                >

                  {/* アップロード動画再生 (bboxモードではポインターイベントをコンテナへ伝達) */}
                  <video
                    ref={videoElRef}
                    className="real-video"
                    src={api.streamUrl(selectedVideo.id)}
                    poster={api.thumbnailUrl(selectedVideo.id)}
                    preload="auto"
                    playsInline
                    controls={false}
                    draggable={false}
                    onDragStart={(e) => e.preventDefault()}

                    onPlay={(e) => {
                      // 検出中の場合は外部メディアキー操作でも再生を即時停止
                      if (isDetectingVehicles) {
                        e.currentTarget.pause();
                        return;
                      }
                      // 検出ボックスは静止画基準のため再生時は自動検出モードを終了(選択ボックスは維持)
                      if (selectionMode === "auto") handleEndDesignateMode();
                      setIsPlaying(true);
                    }}
                    onSeeking={() => {
                      // シーク時も同様に自動検出モードを終了
                      if (selectionMode === "auto" && !isDetectingVehicles) handleEndDesignateMode();
                    }}
                    onPause={() => setIsPlaying(false)}
                    onTimeUpdate={(e) => {
                      if (!isSeekingRef.current) setCurrentTime(e.currentTarget.currentTime);
                    }}
                    onSeeked={handleSeeked}
                    onLoadedMetadata={(e) => {
                      const d = e.currentTarget.duration;
                      setMediaDuration(Number.isFinite(d) ? d : 0);
                    }}
                    onDurationChange={(e) => {
                      const d = e.currentTarget.duration;
                      setMediaDuration(Number.isFinite(d) ? d : 0);
                    }}
                    style={{
                      display: "block",
                      width: "100%",
                      height: "100%",
                      objectFit: "contain",
                      background: "#000",
                      pointerEvents: isBBoxMode ? "none" : "auto",
                    }}
                  />

                  {/* CAMクリップを元動画の時刻に合わせて動画描画領域上に重畳表示。 */}
                  {camShown.map((ev) => (
                    <div
                      key={ev.id}
                      data-event-id={ev.id}
                      className="cam-clip-layer"
                      style={renderGeom ? {
                        left: renderGeom.offsetX,
                        top: renderGeom.offsetY,
                        width: renderGeom.vw * renderGeom.scale,
                        height: renderGeom.vh * renderGeom.scale,
                      } : { inset: 0 }}
                      aria-hidden="true"
                    >
                      <video src={api.clipUrl(ev.id)} muted playsInline preload="auto" draggable={false} onDragStart={(e) => e.preventDefault()} />
                    </div>
                  ))}

                  {/* 車両自動検出中: 動画画面上にローディング表示 */}
                  {isDetectingVehicles && (
                    <div className="vehicle-detect-loading" role="status" aria-live="polite">
                      <div className="vehicle-detect-spinner" />
                      <span>車両を検出中…</span>
                    </div>
                  )}

                  {/* 保存されたバウンディングボックスの描画 (単一ボックス) */}
                  {bboxList.map((box) => {
                    const d = toDispBox(box); // 元座標 → 現在の動画領域の位置(サイズ変更時も追従して再描画)
                    return d && (
                    <div
                      key={box.id}
                      className={`bbox-drawn ${isBBoxMode ? "editing" : ""}`}
                      style={{ left: d.left, top: d.top, width: d.width, height: d.height }}
                      onDragStart={(e) => e.preventDefault()}
                    >
                      {/* 削除(✕): 手動指定中は常時表示、それ以外(車両検知後のモード終了後)はボックスホバー時に表示。
                          分析中は非表示 — 分析は開始時のボックスで実行されるため */}
                      {!isAnalyzing && (
                        <button
                          className="bbox-delete-btn"
                          onPointerDown={(e) => {
                            if (e.isPrimary && e.button === 0) {
                              bboxDeletePointerRef.current = { id: e.pointerId, x: e.clientX, y: e.clientY, moved: false };
                            }
                          }}
                          onPointerMove={(e) => {
                            const gesture = bboxDeletePointerRef.current;
                            if (gesture?.id === e.pointerId && Math.hypot(e.clientX - gesture.x, e.clientY - gesture.y) > 5) {
                              gesture.moved = true;
                            }
                          }}
                          onPointerCancel={() => { bboxDeletePointerRef.current = null; }}
                          onClick={(e) => {
                            e.stopPropagation();
                            const gesture = bboxDeletePointerRef.current;
                            bboxDeletePointerRef.current = null;
                            if (e.detail !== 0 && gesture && (gesture.moved || Math.hypot(e.clientX - gesture.x, e.clientY - gesture.y) > 5)) return;
                            setBboxList((prev) => prev.filter((b) => b.id !== box.id));
                          }}
                          title="指定したボックスを削除します"
                        >
                          ✕
                        </button>
                      )}
                    </div>
                    );
                  })}

                  {/* 現在の手動ドラッグ中ボックスのプレビュー */}
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

                  {/* YOLO自動検出モード: 検出された全bboxを常時表示 */}
                  {selectionMode === "auto" && detectedBoxes.length > 0 && renderGeom && (() => {
                    const { scale, offsetX, offsetY } = renderGeom;
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
                        </div>
                      );
                    });
                  })()}

                  {/* YOLO自動検出モード: マウスホバープレビュー */}
                  {selectionMode === "auto" && hoveredDetectedBox && renderGeom && (() => {
                    const { scale, offsetX, offsetY } = renderGeom;
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
                          🚗 クリックしてこの車両を選択
                        </span>
                      </div>
                    );
                  })()}

                  {videoClickFeedback && !isBBoxMode && !isDetectingVehicles && (
                    <div className="video-click-feedback-layer" aria-hidden="true">
                      <div
                        key={videoClickFeedback.id}
                        className="video-click-feedback"
                      >
                        <img
                          src={VIDEO_FEEDBACK_ICONS[videoClickFeedback.action]}
                          alt=""
                          aria-hidden="true"
                        />
                      </div>
                    </div>
                  )}

                  {showVideoTapControls && !isBBoxMode && !isDetectingVehicles && (
                    <div
                      className="video-tap-controls"
                      role="group"
                      aria-label="動画クイック操作"
                      onClick={(e) => {
                        if (usesTouchVideoControls()) {
                          e.stopPropagation();
                          dismissVideoTapControls();
                        }
                      }}
                    >
                      <button
                        type="button"
                        className="video-tap-skip"
                        aria-label="5秒前へ移動"
                        onClick={(e) => { e.stopPropagation(); seekBySeconds(-5); }}
                      >
                        <img src={replay5Icon} alt="" aria-hidden="true" /><small>5秒前</small>
                      </button>
                      <button
                        type="button"
                        className="video-tap-play"
                        aria-label={isPlaying ? "一時停止" : "再生"}
                        onClick={(e) => { e.stopPropagation(); togglePlay(); revealVideoTapControls(); }}
                      >
                        <PlayerIcon name={isPlaying ? "pause" : "play"} size={30} />
                      </button>
                      <button
                        type="button"
                        className="video-tap-skip"
                        aria-label="5秒後へ移動"
                        onClick={(e) => { e.stopPropagation(); seekBySeconds(5); }}
                      >
                        <img src={forward5Icon} alt="" aria-hidden="true" /><small>5秒後</small>
                      </button>
                    </div>
                  )}
                </div>

                {/* 再生バー + コントロール */}
                <div className="custom-progress-bar">
                  <div
                    className={`progress-track ${isSeeking && didDragSeekRef.current ? "seeking" : ""} ${isSeeking && !didDragSeekRef.current ? "click-seeking" : ""} ${isDetectingVehicles ? "locked" : ""} ${isCurrentTimeInEvent ? "event-current" : ""}`}
                    title={isDetectingVehicles ? "車両検出中はシークできません" : undefined}
                    onPointerDown={handleSeekStart}
                    onPointerMove={handleSeekMove}
                    onPointerUp={handleSeekEnd}
                    onPointerCancel={handleSeekCancel}
                  >
                    <div className="progress-rail">
                      {/* 再生進捗 */}
                      <div
                        className="progress-fill"
                        style={{ width: `${playerDuration ? Math.min(100, (currentTime / playerDuration) * 100) : 0}%` }}
                      />

                      {/* 事故疑い区間 (赤色区間) — クリックで該当時点から再生 */}
                      {playerDuration > 0 &&
                        selectedVideo.events.map((event) => {
                          const start = event.startSec ?? event.timestamp;
                          const end = Math.max(start, event.endSec ?? event.endTimestamp ?? start);
                          const left = Math.min(100, (start / playerDuration) * 100);
                          const width = Math.min(100 - left, ((end - start) / playerDuration) * 100);
                          return (
                            <div
                              key={event.id}
                              className={`event-range ${currentEventId === event.id ? "active" : ""}`}
                              data-event-id={event.id}
                              style={{ left: `${left}%`, width: `${width}%` }}
                              title={`${event.title} ${formatTime(Math.floor(start))} ~ ${formatTime(Math.ceil(end))} · ドラッグして移動`}
                            />
                          );
                        })}
                    </div>
                    <div
                      className="progress-thumb"
                      style={{ left: `${playerDuration ? Math.min(100, (currentTime / playerDuration) * 100) : 0}%` }}
                    />
                    <div className="progress-hover-time" ref={seekHoverRef} />
                  </div>

                  <div className="player-controls">
                    <button
                      className="control-btn"
                      onClick={toggleDesktopVideoPlayback}
                      disabled={isDetectingVehicles}
                      aria-label={isPlaying ? "一時停止" : "再生"}
                      title={isDetectingVehicles ? "車両検出中は再生できません"
                        : isPlaying ? "一時停止します" : "再生します"}
                    >
                      <PlayerIcon name={isPlaying ? "pause" : "play"} size={20} />
                    </button>

                    <div className="volume-control">
                      <button
                        className="control-btn"
                        onClick={toggleMute}
                        aria-label={isMuted || volume === 0 ? "ミュート解除" : "ミュート"}
                        title={isMuted || volume === 0 ? "ミュートを解除します" : "消音します"}
                      >
                        <PlayerIcon
                          name={isMuted || volume === 0 ? "volumeMute" : volume < 50 ? "volumeLow" : "volumeHigh"}
                        />
                      </button>
                      <input
                        className="volume-slider"
                        type="range"
                        min="0"
                        max="100"
                        value={isMuted ? 0 : volume}
                        style={{ "--fill": `${isMuted ? 0 : volume}%` }}
                        onChange={(e) => {
                          const next = Number(e.target.value);
                          const previous = isMuted ? 0 : volume;
                          setVolume(next);
                          setIsMuted(next === 0);
                          showPlayerFeedback(next === 0 ? "volumeOff" : next >= previous ? "volumeUp" : "volumeDown");
                        }}
                        aria-label="音量"
                      />
                    </div>

                    <span className="time-display">
                      {formatTime(Math.floor(currentTime))}
                      <span className="time-sep">/</span>
                      {formatTime(Math.floor(playerDuration))}
                    </span>

                    <div className="right-controls">
                      <span className="cam-now-chip" ref={camChipRef} hidden />
                      <div className="player-settings" ref={playerSettingsRef}>
                        {useCompactFullscreenSettings && (
                          <button
                            type="button"
                            className="control-btn player-settings-trigger"
                            aria-label="再生設定"
                            aria-expanded={isPlayerSettingsOpen}
                            aria-controls="player-settings-options"
                            onClick={() => setIsPlayerSettingsOpen((open) => !open)}
                          >
                            <PlayerIcon name="settings" size={20} />
                          </button>
                        )}
                        <div
                          id="player-settings-options"
                          className={`player-settings-options ${useCompactFullscreenSettings ? "is-popup" : ""}`}
                          role="group"
                          aria-label="再生設定"
                          hidden={useCompactFullscreenSettings && !isPlayerSettingsOpen}
                        >
                        {useCompactFullscreenSettings && <span className="player-setting-label">再生速度</span>}
                        <select
                        className="speed-select"
                        value={playbackSpeed}
                        onChange={(e) => {
                          setPlaybackSpeed(e.target.value);
                          showPlayerFeedback("speed");
                        }}
                        aria-label="再生速度"
                        title="再生速度を変更します"
                      >
                        <option value="0.25">0.25x</option>
                        <option value="0.5">0.5x</option>
                        <option value="1">1x</option>
                        <option value="1.5">1.5x</option>
                        <option value="2">2x</option>
                      </select>
                      {selectedVideo.events.some((e) => e.hasClip && e.clipStartSec != null) && (
                        <>
                        {useCompactFullscreenSettings && <span className="player-setting-label">画面表示</span>}
                        <div className="clip-view-toggle" role="group" aria-label="事故区間画面">
                          <button
                            type="button"
                            className={clipView === "orig" ? "active" : ""}
                            aria-pressed={clipView === "orig"}
                            onClick={() => setClipView("orig")}
                            title="赤色区間でもオリジナル動画を表示します"
                          >原本</button>
                          <button
                            type="button"
                            className={clipView === "cam" ? "active" : ""}
                            aria-pressed={clipView === "cam"}
                            onClick={() => setClipView("cam")}
                            title="事故区間でAIが衝突と判定した箇所を表示します"
                          >衝突表示</button>
                        </div>
                        </>
                      )}
                        </div>
                      </div>
                      {/* 全画面時はシアターモードを非表示 */}
                      {!isFullscreen && (
                        <button
                          className={`control-btn ${isTheaterMode ? "active" : ""}`}
                          onClick={() => {
                            setIsTheaterMode((enabled) => !enabled);
                            showPlayerFeedback("theater");
                          }}
                          disabled={isChoosingVehicle}
                          aria-label="シアターモード"
                          aria-pressed={isTheaterMode}
                          title={isChoosingVehicle ? "車両の選択中は利用できません"
                            : isTheaterMode ? "シアターモードを終了します" : "シアターモードで大きく表示します"}
                        >
                          <PlayerIcon name="theater" />
                        </button>
                      )}
                      <button
                        className="control-btn"
                        onClick={() => {
                          showPlayerFeedback(isFullscreen ? "fullscreenExit" : "fullscreen");
                          handleToggleFullscreen();
                        }}
                        disabled={isChoosingVehicle && !isFullscreen}
                        aria-label={isFullscreen ? "全画面終了" : "全画面"}
                        title={isChoosingVehicle && !isFullscreen ? "車両の選択中は利用できません"
                          : isFullscreen ? "全画面表示を終了します" : "全画面で表示します"}
                      >
                        <PlayerIcon name={isFullscreen ? "exitFullscreen" : "fullscreen"} />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            {/* 右パネル: (1280px以上) 進行ステップ + この動画の分析履歴。幅広画面で閉じた場合(ヘッダーの「パネルを閉じる」)ペイン幅が
                0.3秒かけて0に縮小してフェードアウト(App.css) — その間表示されるようDOMからは除外せず、inert 属性により
                不可視ボタンへのTab・クリックを無効化。狭幅画面では動画下部に常時表示(動画サイズ非依存) */}
            <div className="watch-right" inert={isXLScreen && !isHistoryOpen} aria-hidden={isXLScreen && !isHistoryOpen ? true : undefined}>
              {isXLScreen && workflowSteps}
              <AnalysisHistoryPanel
                history={selectedVideo.analysisHistory || []}
                events={selectedVideo.events}
                progressText={(taskId) => {
                  const job = analyzingJobs.find((j) => j.taskId === taskId);
                  return job ? `分析中 · ${jobProgress(job).shortText}` : null;
                }}
                activeNo={highlightAnalysisNo}
                clearingHistory={clearingHistory}
                onClearHistory={handleClearAnalysisHistory}
                onSelect={(history) => {
                  // 該当分析のクリップを左側リストで強調表示(他はフェードアウト)。同一履歴を再クリックで解除
                  setHighlightAnalysisNo((cur) => (cur === history.no ? null : history.no));
                }}
              />
            </div>
          </div>
        </main>
      )}

      {!selectedVideo && <TeamFooter />}
      {/* 設定モーダル */}
      {isSettingsOpen && (
        <div className="settings-modal-overlay" onClick={() => setIsSettingsOpen(false)}>
          <div className="settings-modal-container" role="dialog" aria-modal="true" aria-label="アカウント設定" onClick={(e) => e.stopPropagation()}>
            <div className="settings-modal-sidebar">
              <span className="account-settings-kicker" translate="no">PERSONAL SPACE</span><h3>アカウント設定</h3><div className="settings-identity"><span>{avatarInitial}</span><strong>{displayName}</strong><small>{roleLabel}</small></div>
              <button
                className={`settings-tab-btn ${activeSettingsTab === "account" ? "active" : ""}`}
                onClick={() => setActiveSettingsTab("account")}
              >
                <InterfaceIcon name="user" /> プロフィール設定
              </button>
              <button
                className={`settings-tab-btn ${activeSettingsTab === "security" ? "active" : ""}`}
                onClick={() => setActiveSettingsTab("security")}
              >
                <InterfaceIcon name="lock" /> パスワード変更
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
                  <span className="account-settings-kicker" translate="no">MY ACCOUNT</span>
                  <h2>{currentUser?.role === "ADMIN" ? "管理者情報設定" : "プロフィール設定"}</h2>
                  <p className="tab-description">
                    {currentUser?.role === "ADMIN"
                      ? "管理者基本情報の確認および修正ができます。"
                      : "ユーザー基本情報の確認および修正ができます。"}
                  </p>

                  <div className="settings-form-group">
                    <label htmlFor="account-username">アカウントID</label>
                    <input id="account-username" type="text" value={currentUsername} disabled className="disabled-input" />
                  </div>

                  <div className="settings-form-group">
                    <label htmlFor="account-name">お名前</label>
                    <input id="account-name"
                      type="text"
                      value={profileName}
                      onChange={(e) => setProfileName(e.target.value)}
                      placeholder="お名前を入力"
                    />
                  </div>

                  <div className="settings-form-group">
                    <label htmlFor="account-phone">連絡先</label>
                    <input id="account-phone"
                      type="text"
                      value={profilePhone}
                      onChange={(e) => setProfilePhone(e.target.value)}
                      placeholder="連絡先を入力 (例: 010-1234-5678)"
                    />
                  </div>

                  <div className="settings-form-group">
                    <label htmlFor="account-email">メールアドレス</label>
                    <input id="account-email"
                      type="email"
                      value={profileEmail}
                      onChange={(e) => setProfileEmail(e.target.value)}
                      placeholder="メールアドレスを入力"
                    />
                  </div>

                  <button
                    className="settings-save-btn"
                    onClick={async () => {
                      if (!profileName.trim()) {
                        alert("お名前を入力してください。");
                        return;
                      }
                      try {
                        const updated = await api.updateMe({
                          name: profileName.trim(),
                          phone: profilePhone.trim(),
                          email: profileEmail.trim(),
                        });
                        if (onUpdateUser) onUpdateUser(updated);
                        showToast("ユーザー情報を正常に保存しました。", "success");
                        alert("ユーザー情報を正常に保存しました。");
                      } catch (err) {
                        alert(`情報修正に失敗しました: ${err.message}`);
                      }
                    }}
                  >
                    変更内容を保存
                  </button>
                </div>
              )}

              {activeSettingsTab === "security" && (
                <div className="settings-tab-content">
                  <span className="account-settings-kicker" translate="no">MY ACCOUNT</span>
                  <h2>パスワード変更</h2>
                  <p className="tab-description">システムのセキュリティ保護のため、定期的にパスワードを変更してください。</p>

                  <div className="settings-form-group">
                    <label htmlFor="account-current-password">現在のパスワード</label>
                    <input id="account-current-password"
                      type="password"
                      placeholder="現在のパスワードを入力"
                      value={currentPassword}
                      onChange={(e) => setCurrentPassword(e.target.value)}
                    />
                  </div>

                  <div className="settings-form-group">
                    <label htmlFor="account-new-password">新しいパスワード</label>
                    <input id="account-new-password"
                      type="password"
                      placeholder="新しいパスワードを入力"
                      value={newPassword}
                      onChange={(e) => setNewPassword(e.target.value)}
                    />
                  </div>

                  <div className="settings-form-group">
                    <label htmlFor="account-confirm-password">新しいパスワード（確認）</label>
                    <input id="account-confirm-password"
                      type="password"
                      placeholder="新しいパスワードを再入力"
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
                        showToast("パスワードが正常に変更されました。", "success");
                        alert("パスワードが正常に変更されました。");
                        setCurrentPassword("");
                        setNewPassword("");
                        setConfirmPassword("");
                        setIsSettingsOpen(false);
                      } catch (err) {
                        alert(`パスワード変更に失敗しました: ${err.message}`);
                      }
                    }}
                  >
                    パスワード変更を完了
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* トースト通知 */}
      {toast && (
        <div className={`app-toast app-toast-${toast.type}`}>{toast.message}</div>
      )}

      {/* 動画情報ポップアップ */}
      {infoVideo && (() => {
        const v = infoVideo;
        const up = formatAnalyzedAt(v.uploadedAt);
        const lastStatus = v.lastTask ? (HISTORY_STATUS[v.lastTask.status]?.label || v.lastTask.status) : null;
        const rows = [
          ["録画日", v.date || "-"],
          ["アップロード日時", up ? `${up[0]} ${up[1]}` : "-"],
          ["元ファイル名", v.fileName || "-"],
          ["動画の長さ", formatTime(v.duration)],
          ["解像度", v.width && v.height ? `${v.width}×${v.height}${v.fps ? ` · ${Math.round(v.fps)}fps` : ""}` : "-"],
          ["事故疑いクリップ", `${v.events.length}件`],
          ["最終分析", lastStatus || "未実施"],
        ];
        return (
          <div className="settings-modal-overlay" onClick={() => setInfoVideo(null)}>
            <div
              className="video-info-modal"
              role="dialog"
              aria-modal="true"
              aria-labelledby="video-info-title"
              onClick={(e) => e.stopPropagation()}
              onKeyDown={(e) => {
                if (e.key === "Escape") setInfoVideo(null);
              }}
            >
              <div className="video-info-modal-header">
                <h2 id="video-info-title" title={getVideoDisplayTitle(v, videos)}>
                  {getVideoDisplayTitle(v, videos)}
                </h2>
                <button className="clip-modal-close" onClick={() => setInfoVideo(null)} autoFocus aria-label="閉じる">
                  ✕
                </button>
              </div>
              <dl className="video-info-list">
                {rows.map(([k, val]) => (
                  <div key={k}>
                    <dt>{k}</dt>
                    <dd title={val}>{val}</dd>
                  </div>
                ))}
              </dl>
            </div>
          </div>
        );
      })()}

      {/* 事故クリップ削除確認 (1件 · 一括) */}
      {(pendingDeleteEvent || pendingClearAll) && (() => {
        const isAll = pendingClearAll;
        const closeConfirm = () => {
          setPendingDeleteEvent(null);
          setPendingClearAll(false);
        };
        return (
          <div className="settings-modal-overlay" onClick={closeConfirm}>
            <div
              className="confirm-modal"
              role="alertdialog"
              aria-modal="true"
              aria-labelledby="confirm-modal-title"
              aria-describedby="confirm-modal-desc"
              onClick={(e) => e.stopPropagation()}
              onKeyDown={(e) => {
                if (e.key === "Escape") closeConfirm();
              }}
            >
              <div className="confirm-modal-icon" aria-hidden="true">
                <TrashIcon size={24} />
              </div>
              <h2 id="confirm-modal-title" className="confirm-modal-title">
                {isAll
                  ? `事故疑いクリップ ${selectedVideo?.events.length ?? 0}件をすべて削除しますか？`
                  : "クリップ動画を削除しますか？"}
              </h2>
              <p id="confirm-modal-desc" className="confirm-modal-desc">
                {isAll
                  ? "この動画で検出されたすべてのクリップ動画および検出ログが削除されます。"
                  : "該当する事故区間のクリップ動画および検出ログが削除されます。"}
                <br />
                削除後は元に戻すことができません。
              </p>
              <div className="upload-modal-actions">
                <button
                  className="confirm-danger-btn"
                  onClick={() => {
                    const id = pendingDeleteEvent?.id;
                    closeConfirm();
                    if (isAll) handleClearAllEvents();
                    else handleDeleteEvent(id);
                  }}
                >
                  {isAll ? "すべて削除" : "削除"}
                </button>
                <button
                  className="upload-cancel-btn"
                  autoFocus
                  onClick={closeConfirm}
                >
                  キャンセル
                </button>
              </div>
            </div>
          </div>
        );
      })()}

      {/* 事故区間クリップポップアップ (+ オリジナルクリップダウンロード) */}
      {clipEvent && (
        <div className="settings-modal-overlay clip-viewer-overlay" onClick={() => setClipEvent(null)}>
          <div
            className="clip-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="clip-modal-title"
            onClick={(e) => e.stopPropagation()}
            onKeyDown={(e) => {
              if (e.key === "Escape") setClipEvent(null);
            }}
          >
            <div className="clip-modal-header">
              <div className="clip-modal-heading">
                <h2 id="clip-modal-title" className="clip-modal-title">事故区間クリップ</h2>
                <div className="clip-modal-chips">
                  <span className="clip-chip">
                    ⏱ {formatTime(clipEvent.timestamp)}
                    {clipEvent.endTimestamp != null ? ` ~ ${formatTime(clipEvent.endTimestamp)}` : ""}
                  </span>
                  {clipEvent.prob != null && (
                    <span className="clip-chip clip-chip-danger">
                      事故確率 {(clipEvent.prob * 100).toFixed(1)}%
                    </span>
                  )}
                </div>
              </div>
              <button
                className="clip-close-btn"
                onClick={() => setClipEvent(null)}
                aria-label="クリップを閉じる"
                autoFocus
              >
                ✕
              </button>
            </div>

            <div className="clip-video-frame">
              <video className="clip-modal-video" src={api.clipUrl(clipEvent.id)} controls autoPlay draggable={false} onDragStart={(e) => e.preventDefault()} />
            </div>

            <div className="clip-modal-footer">
              <div className="clip-legend">
                <span className="clip-legend-item">
                  <i className="clip-legend-swatch clip-legend-a" /> 赤枠 — 事故が疑われる瞬間です
                </span>
                <span className="clip-legend-item">
                  <i className="clip-legend-swatch clip-legend-s" /> 緑枠 — 特段の異常が見られない場面です
                </span>
              </div>
              {clipEvent.hasRawClip ? (
                <a
                  className="clip-download-btn"
                  href={api.rawClipUrl(clipEvent.id)}
                  download
                  title="バウンディングボックスやステータス表示のない元の事故区間動画をダウンロードします"
                >
                  ⬇ オリジナルクリップをダウンロード
                </a>
              ) : (
                <span className="clip-download-unavailable" title="この動画を再分析するとオリジナルクリップが生成されます">
                  オリジナルクリップなし · 再分析時に生成
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 動画アップロードモーダル */}
      {showUpload && (
        <UploadModal
          onClose={() => setShowUpload(false)}
          onUploaded={async () => {
            setShowUpload(false);
            showToast("アップロード完了", "success");
            await loadVideos();
          }}
          onError={(msg) => showToast(msg, "error")}
        />
      )}
      {/* 同一日付の複数動画選択モーダル */}
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
      title: "", // 任意 — 空欄の場合は「録画日 録画データ #N」
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

  const updateFileTitle = (id, title) => {
    setFilesList((prev) => prev.map((item) => (item.id === id ? { ...item, title } : item)));
  };

  const updateAllDates = (dateStr) => {
    setFilesList((prev) => prev.map((item) => ({ ...item, recordingDate: dateStr })));
  };

  const handleUpload = async () => {
    if (filesList.length === 0) {
      onError("アップロードする動画ファイルを選択してください。");
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
        await api.uploadVideo(item.file, item.recordingDate || null, item.title);
        successCount += 1;
      } catch (e) {
        failCount += 1;
        console.error(`Failed to upload ${item.file.name}:`, e);
      }
    }

    setUploading(false);
    if (successCount > 0) {
      if (failCount > 0) {
        onError(`${successCount}件のアップロード完了 (${failCount}件失敗)`);
      }
      onUploaded();
    } else {
      onError("すべての動画ファイルのアップロードに失敗しました。");
    }
  };

  return (
    <div className="settings-modal-overlay" onClick={onClose}>
      <div className="upload-modal multi-upload-modal" onClick={(e) => e.stopPropagation()}>
        <div className="upload-modal-header">
          <div>
            <h2>動画の一括アップロード</h2>
            <p className="tab-description">分析対象の防犯カメラ録画映像をアップロードします。複数ファイルを一度に選択し、ファイルごとに録画日やタイトルを設定できます。</p>
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
            <p className="dropzone-title">クリックして動画ファイルを選択 (複数選択可能)</p>
            <p className="dropzone-sub">またはここに動画ファイルをドラッグ＆ドロップしてください</p>
          </div>
        ) : (
          <div className="upload-file-list-container">
            <div className="upload-file-list-toolbar">
              <span className="file-count-badge">選択された動画: <strong>{filesList.length}件</strong></span>
              {/* ファイル追加はリスト下のドロップエリアで行う */}
              <div className="file-toolbar-actions">
                <button
                  type="button"
                  className="clear-all-files-btn"
                  onClick={() => setFilesList([])}
                  disabled={uploading}
                >
                  一括削除
                </button>
              </div>
            </div>

            <div className="bulk-date-bar">
              <span>一括録画日指定:</span>
              <input
                type="date"
                max={formatDate(new Date())}
                onChange={(e) => {
                  if (e.target.value) updateAllDates(e.target.value);
                }}
                disabled={uploading}
              />
              <span className="bulk-date-tip">(ファイルごとの個別設定も可能)</span>
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
                    <label>録画日:</label>
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
                    title="このファイルをアップロードリストから除外します"
                  >
                    ✕
                  </button>

                  {/* タイトル(任意) */}
                  <div className="file-item-title-group">
                    <label htmlFor={`upload-title-${item.id}`}>タイトル:</label>
                    <input
                      id={`upload-title-${item.id}`}
                      type="text"
                      value={item.title}
                      maxLength={100}
                      placeholder={`未入力の場合は「${item.recordingDate || "録画日"} 録画データ」と表示されます`}
                      onChange={(e) => updateFileTitle(item.id, e.target.value)}
                      disabled={uploading}
                    />
                  </div>
                </div>
              ))}
            </div>

            {/* 追加ドロップエリア */}
            <div
              className={`upload-dropzone upload-dropzone-compact ${dragActive ? "drag-active" : ""} ${uploading ? "disabled" : ""}`}
              onDragEnter={uploading ? undefined : handleDrag}
              onDragOver={uploading ? undefined : handleDrag}
              onDragLeave={uploading ? undefined : handleDrag}
              onDrop={uploading ? undefined : handleDrop}
              onClick={() => {
                if (!uploading) fileInputRef.current?.click();
              }}
            >
              <span className="dropzone-compact-icon" aria-hidden="true">＋</span>
              <span>ここに動画ファイルをドラッグ＆ドロップするか、クリックして追加</span>
            </div>
          </div>
        )}

        {uploading && (
          <div className="upload-progress-box">
            <div className="upload-progress-info">
              <span>アップロード中 ({uploadProgress.current}/{uploadProgress.total})</span>
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
              ? `アップロード中 (${uploadProgress.current}/${uploadProgress.total})...`
              : `${filesList.length > 0 ? `${filesList.length}件の` : ""}動画をアップロード`}
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
            <h2>📅 {dateText} 動画リスト ({videos.length}件)</h2>
            <p className="tab-description">
              該当する日付に合計 {videos.length}件の動画があります。再生する動画を選択してください。
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
                  alt={`${v.date} サムネイル`}
                  onError={(e) => { e.currentTarget.style.display = "none"; }}
                />
                <span className="event-count-badge">疑い区間 {v.events.length}件</span>
              </div>
              <div className="date-video-card-info">
                <h3>{getVideoDisplayTitle(v, videos)}</h3>
                <div className="date-video-card-meta">
                  <span>⏱ 動画の長さ: {formatTime(v.duration)}</span>
                </div>
              </div>
              <button className="date-video-play-btn">
                ▶ 再生する
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
  const [isDarkMode, setIsDarkMode] = useState(() => localStorage.getItem("siot_dark_mode") === "true");
  // ログイン済みの場合は描画前にセッションを開始 — 動画・サムネイル・クリップはこの時点で取得する
  // クッキーを介して開かれるため(ヘッダーを付与できない<video>・<img>)、先に描画するとサムネイル取得がブロックされる
  const [sessionReady, setSessionReady] = useState(() => !getToken());

  useEffect(() => {
    document.body.classList.toggle("dark-mode", isDarkMode);
    localStorage.setItem("siot_dark_mode", String(isDarkMode));
  }, [isDarkMode]);

  useEffect(() => {
    if (!getToken()) return;
    api.startSession().then((u) => {
      setCurrentUser(u);
      saveAuth(getToken(), u);
    }).catch(() => { })   // 期限切れ(401)の場合はapi.jsがログイン画面へリダイレクト
      .finally(() => setSessionReady(true));
  }, []);

  const handleUpdateUser = (updatedUser) => {
    setCurrentUser(updatedUser);
    saveAuth(getToken(), updatedUser);
  };

  const handleLogout = () => {
    api.logout().catch(() => { });   // 動画クッキーも削除
    clearAuth();
    setCurrentUser(null);
    setUser(false);
  };

  if (!sessionReady) return null;

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
                isDarkMode={isDarkMode}
                setIsDarkMode={setIsDarkMode}
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
                isDarkMode={isDarkMode}
                setIsDarkMode={setIsDarkMode}
                onUpdateUser={handleUpdateUser}
                onLogout={handleLogout}
                view="watch"
              />
            </RequireAuth>
          }
        />
        <Route path="*" element={<Navigate to={user ? "/videos" : "/login"} replace />} />
      </Routes>
    </BrowserRouter>
  );
}
