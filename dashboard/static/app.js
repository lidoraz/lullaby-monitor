/* ============================================================
   Lullaby Monitor — app.js
   ============================================================ */

const API = '';   // same-origin

// ---- Event-type display config ----
const EVENT_META = {
    baby_cry: {
        label: '👶 Baby cry',
        cls: 'baby_cry',
        tooltip: 'Infant or baby crying detected by audio analysis',
    },
    yell: {
        label: '🗣 Yell',
        cls: 'yell',
        tooltip: 'Shouting, screaming, or loud angry voice detected',
    },
    loud_noise: {
        label: '💥 Loud noise',
        cls: 'loud_noise',
        tooltip: 'Sudden loud sound such as crash, bang, or impact',
    },
    abuse: {
        label: '⚠️ Abuse alert',
        cls: 'abuse',
        tooltip: 'Baby crying + caregiver yelling within 3 seconds (potential crisis)',
    },
    talk: {
        label: '💬 Talk',
        cls: 'talk',
        tooltip: 'Speech or conversation detected (caregiver present)',
    },
};

// seconds in a full day — timeline x-axis spans this
const DAY_SECS = 86400;

// ============================================================
// Filter state
// ============================================================
const filters = {
    sev: new Set(['HIGH', 'MEDIUM', 'LOW']),
    type: new Set(['baby_cry', 'abuse']),
    minConf: 0.5,   // 0–1 (50% by default)
};

function eventPassesFilter(ev) {
    if (!filters.sev.has(ev.severity)) return false;
    if (!filters.type.has(ev.event_type)) return false;
    if ((ev.peak_conf ?? ev.peak_confidence ?? 0) < filters.minConf) return false;
    return true;
}

// ============================================================
// State
// ============================================================
let _autoFollow = true;   // keep timeline centred on playhead by default
let _inAutoFollow = false; // re-entrancy guard – prevents setView→redrawTimeline→updatePlayhead loop

let currentDate = null;
let currentRecs = [];   // array of recording objects for selected date
let evtSource = null; // SSE EventSource

// Skip-silence player state
let _skipSilence = false;
let _playerSilentRegions = [];   // [[start_s, end_s], …] for current recording
let _activeRec = null;           // recording currently loaded in the player

// Zoom / pan view state  — [start, end] in seconds since midnight
const view = { start: 0, end: DAY_SECS };
let _zoomPanBound = false;

// Autoplay queue state
let _autoplayQueue = [];   // [{ rec, ev }] sorted chronologically
let _autoplayIdx = -1;   // -1 = not in autoplay mode
let _autoplayPostBuffer = 5; // seconds to keep playing past offset_end before advancing

// ============================================================
// DOM refs
// ============================================================
const elDateList = document.getElementById('date-list');
const elWelcome = document.getElementById('welcome');
const elTimelineView = document.getElementById('timeline-view');
const elTlDateTitle = document.getElementById('tl-date-title');
const elHourRuler = document.getElementById('hour-ruler');
const elTlViewport = document.getElementById('tl-viewport');
const elTlCanvas = document.getElementById('tl-canvas');
const elTlViewRange = document.getElementById('tl-view-range');

// Playhead: created once in JS, re-appended to tl-canvas after every redraw
const elTlPlayhead = document.createElement('div');
elTlPlayhead.id = 'tl-playhead';
elTlPlayhead.style.visibility = 'hidden';
elTlCanvas.appendChild(elTlPlayhead);
const elEventsPane = document.getElementById('events-pane');
const elEventList = document.getElementById('event-list');
const elEventCount = document.getElementById('event-count');
const elPlayerDrawer = document.getElementById('player-drawer');
const elVideoEl = document.getElementById('video-el');
const elPlayerTitle = document.getElementById('player-title');
const elPlayerBadge = document.getElementById('player-event-badge');
const elPlayerTimestamp = document.getElementById('player-timestamp');
const elPlayerGlobalTimestamp = document.getElementById('player-global-timestamp');
const elBtnSkipSilence = document.getElementById('btn-skip-silence');
const elBtnFixAudio = document.getElementById('btn-fix-audio');
const elBtnClosePlayer = document.getElementById('btn-close-player');
const elBtnGotoMarker = document.getElementById('btn-goto-marker');
const elBtnAutoFollow = document.getElementById('btn-auto-follow');
const elBtnPlayAll = document.getElementById('btn-play-all');
const elAutoplayBar = document.getElementById('autoplay-bar');
const elAutoplayCounter = document.getElementById('autoplay-counter');
const elBtnAutoplayPrev = document.getElementById('btn-autoplay-prev');
const elBtnAutoplayNext = document.getElementById('btn-autoplay-next');
const elBtnAutoplayStop = document.getElementById('btn-autoplay-stop');
const elInpAutoplayBuffer = document.getElementById('inp-autoplay-buffer');
const elBtnProcess = document.getElementById('btn-process');
const elInpSource = document.getElementById('inp-source');
const elChkForce = document.getElementById('chk-force');
const elProgressWrap = document.getElementById('progress-wrap');
const elProgressFill = document.getElementById('progress-fill');
const elProgressMsg = document.getElementById('progress-msg');
const elStatsCard = document.getElementById('stats-card');
const elStatRecs = document.getElementById('stat-recs');
const elStatEvents = document.getElementById('stat-events');
const elStatDates = document.getElementById('stat-dates');
const elSettingsModal = document.getElementById('settings-modal');
const elBtnSettings = document.getElementById('btn-settings');
const elBtnCloseSettings = document.getElementById('btn-close-settings');
const elBtnSaveSettings = document.getElementById('btn-save-settings');
const elFilterConf = document.getElementById('filter-conf');
const elFilterConfLbl = document.getElementById('filter-conf-lbl');
const elBtnFilterReset = document.getElementById('btn-filter-reset');
const elExportModal = document.getElementById('export-modal');
const elBtnCloseExport = document.getElementById('btn-close-export');
const elExportEventInfo = document.getElementById('export-event-info');
const elExpPre = document.getElementById('exp-pre');
const elExpPost = document.getElementById('exp-post');
const elBtnExportGo = document.getElementById('btn-export-go');
const elExportStatus = document.getElementById('export-status');

// export: holds the event + rec currently being exported
let exportTarget = null;

// ============================================================
// Init
// ============================================================
(async function init() {
    await loadStats();
    await loadDates();
    bindSettings();
    await loadSettingsForm();

    elBtnProcess.addEventListener('click', startProcessing);
    elBtnClosePlayer.addEventListener('click', () => { stopAutoplay(); closePlayer(); });
    bindFilters();
    bindExport();
    bindAutoplay();
})();

// ============================================================
// Export
// ============================================================
function bindExport() {
    elBtnCloseExport.addEventListener('click', closeExportModal);
    elExportModal.addEventListener('click', e => {
        if (e.target === elExportModal) closeExportModal();
    });
    elBtnExportGo.addEventListener('click', runExport);
}

function openExportModal(rec, ev) {
    exportTarget = { rec, ev };
    const meta = EVENT_META[ev.event_type] ?? { label: ev.event_type };
    const absTime = ev.abs_start.slice(0, 19).replace('T', ' ');
    const dur = (ev.offset_end - ev.offset_start).toFixed(1);
    const conf = Math.round((ev.peak_conf ?? 0) * 100);
    elExportEventInfo.innerHTML =
        `<strong>${meta.label}</strong>  &nbsp;${ev.severity}<br/>` +
        `🕐 ${absTime}<br/>` +
        `⏱ ${dur}s duration &nbsp;·&nbsp; 🎯 ${conf}% confidence`;

    elExportStatus.className = 'export-status hidden';
    elExportStatus.textContent = '';
    elBtnExportGo.disabled = false;
    elExportModal.classList.remove('hidden');
}

function closeExportModal() {
    elExportModal.classList.add('hidden');
    exportTarget = null;
}

async function runExport() {
    if (!exportTarget) return;
    const { rec, ev } = exportTarget;
    const pre = parseFloat(elExpPre.value) || 5;
    const post = parseFloat(elExpPost.value) || 20;
    const mode = document.querySelector('input[name="exp-mode"]:checked')?.value ?? 'video';

    setExportStatus('busy', '⏳ Exporting…');
    elBtnExportGo.disabled = true;

    try {
        const result = await fetchJSON('/api/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_path: rec.file_path,
                offset_start: ev.offset_start,
                offset_end: ev.offset_end,
                abs_start: ev.abs_start,
                event_type: ev.event_type,
                pre_seconds: pre,
                post_seconds: post,
                mode,
            }),
        });
        const dlUrl = `/api/export/download?path=${encodeURIComponent(result.path)}`;
        setExportStatus('ok',
            `✓ Saved: <strong>${escHtml(result.filename)}</strong> &nbsp;(${result.size_kb} KB)` +
            `&nbsp;&nbsp;<a href="${dlUrl}" download="${escHtml(result.filename)}" style="color:#86efac">⬇ Download</a>`);
    } catch (e) {
        setExportStatus('error', `✗ ${e.message}`);
    }
    elBtnExportGo.disabled = false;
}

function setExportStatus(cls, html) {
    elExportStatus.className = `export-status ${cls}`;
    elExportStatus.innerHTML = html;
    elExportStatus.classList.remove('hidden');
}

// ============================================================
// Filters
// ============================================================
function bindFilters() {
    // Chip toggles
    document.querySelectorAll('.filter-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const group = chip.dataset.filter;   // 'sev' or 'type'
            const val = chip.dataset.val;
            const set = filters[group];
            if (set.has(val)) {
                // Don't allow deselecting all in a group
                if (set.size > 1) { set.delete(val); chip.classList.remove('active'); }
            } else {
                set.add(val); chip.classList.add('active');
            }
            applyFilters();
        });
    });

    // Confidence slider
    elFilterConf.addEventListener('input', () => {
        const pct = parseInt(elFilterConf.value, 10);
        elFilterConfLbl.textContent = pct + '%';
        filters.minConf = pct / 100;
        applyFilters();
    });

    // Reset button
    elBtnFilterReset.addEventListener('click', () => {
        filters.sev = new Set(['HIGH', 'MEDIUM', 'LOW']);
        filters.type = new Set(['baby_cry', 'yell', 'loud_noise', 'abuse', 'talk']);
        filters.minConf = 0;
        elFilterConf.value = 0;
        elFilterConfLbl.textContent = '0%';
        document.querySelectorAll('.filter-chip').forEach(c => c.classList.add('active'));
        applyFilters();
    });
}

/**
 * Re-apply current filters without reloading data from the server.
 * Updates timeline marker visibility and the event list.
 */
function applyFilters() {
    if (!currentRecs.length) return;
    // Filters changed — any running autoplay queue is now stale
    stopAutoplay();

    // Timeline markers: toggle .filtered-out class
    document.querySelectorAll('.tl-event').forEach(el => {
        const ev = el._eventData;
        if (!ev) return;
        if (eventPassesFilter(ev)) {
            el.classList.remove('filtered-out');
        } else {
            el.classList.add('filtered-out');
        }
    });

    // Re-render event list with current filter
    renderEventList(currentRecs);
}

// ============================================================
// Stats
// ============================================================
async function loadStats() {
    try {
        const stats = await fetchJSON('/api/stats');
        elStatRecs.textContent = stats.total_recordings;
        elStatEvents.textContent = stats.total_events;
        elStatDates.textContent = stats.dates_count;
        elStatsCard.classList.remove('hidden');
    } catch { /* ignore on first run */ }
}

// ============================================================
// Date list
// ============================================================
async function loadDates() {
    try {
        const dates = await fetchJSON('/api/dates');
        renderDateList(dates);
    } catch (e) {
        console.error('Failed to load dates', e);
    }
}

function renderDateList(dates) {
    elDateList.innerHTML = '';
    if (!dates.length) {
        elDateList.innerHTML = '<li style="padding:12px 14px;color:var(--text-dim);font-size:12px">No recordings yet.</li>';
        return;
    }
    dates.forEach(d => {
        const li = document.createElement('li');
        li.dataset.date = d;
        // format nicely: 2026-02-24 → Mon 24 Feb 2026
        const dt = new Date(d + 'T12:00:00');
        const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
        const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        li.innerHTML = `
      <span>${dayNames[dt.getDay()]} ${dt.getDate()} ${monthNames[dt.getMonth()]} ${dt.getFullYear()}</span>
    `;
        li.addEventListener('click', () => selectDate(d));
        elDateList.appendChild(li);
    });
}

// ============================================================
// Select date → load recordings → render timeline
// ============================================================
async function selectDate(date) {
    currentDate = date;
    stopAutoplay();
    closePlayer();

    // Highlight active
    document.querySelectorAll('#date-list li').forEach(li => {
        li.classList.toggle('active', li.dataset.date === date);
    });

    try {
        const recs = await fetchJSON(`/api/date/${date}`);
        currentRecs = recs;
        renderTimeline(date, recs);
    } catch (e) {
        console.error('Failed to load date', e);
    }
}

// ============================================================
// Timeline rendering — unified day canvas with zoom + pan
// ============================================================
function renderTimeline(date, recs) {
    elWelcome.classList.add('hidden');
    elTimelineView.classList.remove('hidden');

    const dt = new Date(date + 'T12:00:00');
    const dayNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    const monthNames = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
    elTlDateTitle.textContent = `${dayNames[dt.getDay()]}, ${monthNames[dt.getMonth()]} ${dt.getDate()} ${dt.getFullYear()}`;

    bindZoomPan();     // no-op after first call
    fitToRecs(recs);   // sets view + triggers redrawTimeline
    renderEventList(recs);
}

// Seconds since midnight from an ISO datetime string
function secsSinceMidnight(isoString) {
    const d = new Date(isoString);
    return d.getHours() * 3600 + d.getMinutes() * 60 + d.getSeconds();
}

// Seconds → "HH:MM"
function formatTimeSecs(secs) {
    const s = ((secs % DAY_SECS) + DAY_SECS) % DAY_SECS;
    return String(Math.floor(s / 3600)).padStart(2, '0') + ':' +
        String(Math.floor((s % 3600) / 60)).padStart(2, '0');
}

// Adaptive hour ruler — tick density scales with the current view span
function renderHourRuler() {
    elHourRuler.innerHTML = '';
    const span = view.end - view.start;
    const tickSecs =
        span <= 600 ? 60 :   // ≤10 min  → 1 min ticks
            span <= 1800 ? 300 :   // ≤30 min  → 5 min ticks
                span <= 7200 ? 600 :   // ≤2 hr    → 10 min ticks
                    span <= 18000 ? 1800 :   // ≤5 hr    → 30 min ticks
                        span <= 43200 ? 3600 :   // ≤12 hr   → 1 hr ticks
                            7200;     // full day → 2 hr ticks

    const first = Math.ceil(view.start / tickSecs) * tickSecs;
    for (let t = first; t <= view.end; t += tickSecs) {
        const pct = (t - view.start) / span * 100;
        if (pct < 0 || pct > 100) continue;
        const tick = document.createElement('div');
        tick.className = 'ruler-tick';
        tick.style.left = pct.toFixed(4) + '%';
        tick.textContent = formatTimeSecs(t);
        elHourRuler.appendChild(tick);
    }
}

// Greedy lane assignment for overlapping recordings
function assignLanes(okRecs) {
    const laneEnd = [];
    return okRecs.map(rec => {
        const s = secsSinceMidnight(rec.rec_start);
        const e = s + rec.duration;
        let lane = laneEnd.findIndex(le => le <= s);
        if (lane === -1) lane = laneEnd.length;
        laneEnd[lane] = e;
        return lane;
    });
}

// Render all recordings as absolute blocks on the unified canvas
function renderCanvas(recs) {
    elTlCanvas.innerHTML = '';
    const span = view.end - view.start;
    const LANE_H = 28, LANE_GAP = 6, PAD_V = 4;

    const okRecs = recs.filter(r => r.status === 'ok');
    if (!okRecs.length) {
        elTlCanvas.appendChild(elTlPlayhead);
        return;
    }

    const lanes = assignLanes(okRecs);
    const numLanes = Math.max(...lanes) + 1;
    elTlViewport.style.height = (PAD_V + numLanes * (LANE_H + LANE_GAP) - LANE_GAP + PAD_V) + 'px';

    okRecs.forEach((rec, i) => {
        const lane = lanes[i];
        const recStart = secsSinceMidnight(rec.rec_start);

        // Skip blocks entirely outside the view window
        if (recStart + rec.duration < view.start || recStart > view.end) return;

        const left = ((recStart - view.start) / span * 100).toFixed(4) + '%';
        const width = (rec.duration / span * 100).toFixed(4) + '%';
        const top = PAD_V + lane * (LANE_H + LANE_GAP);

        const block = document.createElement('div');
        block.className = 'rec-block';
        block.style.cssText = `left:${left};width:${width};top:${top}px;height:${LANE_H}px`;
        const startStr = rec.rec_start.slice(11, 16);
        const endStr = rec.rec_end.slice(11, 16);
        block.dataset.tip = `${startStr}–${endStr} · click to play`;

        // Time label inside block (hidden by overflow when narrow)
        const lbl = document.createElement('span');
        lbl.className = 'rec-block-label';
        lbl.textContent = `${startStr}–${endStr}`;
        block.appendChild(lbl);

        // Silence stripes (relative to recording duration)
        (rec.silent_regions || []).forEach(([s, e]) => {
            const stripe = document.createElement('div');
            stripe.className = 'tl-silence';
            stripe.style.left = (s / rec.duration * 100).toFixed(4) + '%';
            stripe.style.width = ((e - s) / rec.duration * 100).toFixed(4) + '%';
            block.appendChild(stripe);
        });

        // Event markers (relative to recording duration)
        rec.events.forEach(ev => {
            const el = document.createElement('div');
            el.className = `tl-event ${ev.event_type}`;
            if (!eventPassesFilter(ev)) el.classList.add('filtered-out');
            el.style.left = (ev.offset_start / rec.duration * 100).toFixed(4) + '%';
            el.style.width = Math.max((ev.offset_end - ev.offset_start) / rec.duration * 100, 0.3).toFixed(4) + '%';
            const evLabel = EVENT_META[ev.event_type]?.label ?? ev.event_type;
            const evTooltip = EVENT_META[ev.event_type]?.tooltip ?? ev.event_type;
            const confPct = Math.round((ev.peak_conf ?? 0) * 100);
            el.dataset.tip = `${evLabel}  ${fmtOffset(ev.offset_start)}  ${ev.severity}  ${confPct}%`;
            el.title = `${evTooltip}\n${fmtOffset(ev.offset_start)} – ${fmtOffset(ev.offset_end)} (${ev.severity}, ${confPct}%)`;
            el._eventData = ev;
            el.addEventListener('click', e => { e.stopPropagation(); openPlayer(rec, ev.offset_start, ev); });
            block.appendChild(el);
        });

        block.addEventListener('click', () => openPlayer(rec, 0, null));
        elTlCanvas.appendChild(block);
    });

    // Always keep the playhead on top inside the canvas
    elTlCanvas.appendChild(elTlPlayhead);
}

// ---- View management --------------------------------------------------------

function fitToRecs(recs) {
    const ok = recs.filter(r => r.status === 'ok');
    if (!ok.length) { view.start = 0; view.end = DAY_SECS; redrawTimeline(); return; }
    const starts = ok.map(r => secsSinceMidnight(r.rec_start));
    const ends = ok.map(r => secsSinceMidnight(r.rec_start) + r.duration);
    const minS = Math.min(...starts), maxE = Math.max(...ends);
    const pad = Math.max((maxE - minS) * 0.08, 300);
    setView(minS - pad, maxE + pad);
}

function setView(start, end) {
    let s = start;
    const span = Math.max(end - start, 120);
    let e = s + span;
    if (s < 0) { e -= s; s = 0; }
    if (e > DAY_SECS) { s -= (e - DAY_SECS); e = DAY_SECS; s = Math.max(0, s); }
    view.start = s;
    view.end = e;
    redrawTimeline();
}

function zoomTo(center, newSpan) {
    const clamped = Math.max(120, Math.min(DAY_SECS, newSpan));
    setView(center - clamped / 2, center + clamped / 2);
}

function redrawTimeline() {
    renderHourRuler();
    renderCanvas(currentRecs);
    elTlViewRange.textContent = formatTimeSecs(view.start) + ' – ' + formatTimeSecs(view.end);
    updatePlayhead();
}

function bindZoomPan() {
    if (_zoomPanBound) return;
    _zoomPanBound = true;

    // Scroll-wheel zoom centred on mouse cursor
    elTlViewport.addEventListener('wheel', e => {
        e.preventDefault();
        const rect = elTlViewport.getBoundingClientRect();
        const ratio = (e.clientX - rect.left) / rect.width;
        const span = view.end - view.start;
        const center = view.start + ratio * span;
        zoomTo(center, span * (e.deltaY > 0 ? 1.35 : 1 / 1.35));
    }, { passive: false });

    // Drag-to-pan
    let dragging = false, dragX = 0, dragViewStart = 0;
    elTlViewport.addEventListener('mousedown', e => {
        if (e.target.closest('.tl-event')) return;
        dragging = true;
        dragX = e.clientX;
        dragViewStart = view.start;
        elTlViewport.style.cursor = 'grabbing';
        e.preventDefault();
    });
    document.addEventListener('mousemove', e => {
        if (!dragging) return;
        const rect = elTlViewport.getBoundingClientRect();
        const dx = (dragX - e.clientX) / rect.width;
        const span = view.end - view.start;
        setView(dragViewStart + dx * span, dragViewStart + dx * span + span);
    });
    document.addEventListener('mouseup', () => {
        if (dragging) { dragging = false; elTlViewport.style.cursor = ''; }
    });

    // Toolbar buttons
    document.getElementById('btn-zoom-in').addEventListener('click', () =>
        zoomTo((view.start + view.end) / 2, (view.end - view.start) / 2));
    document.getElementById('btn-zoom-out').addEventListener('click', () =>
        zoomTo((view.start + view.end) / 2, (view.end - view.start) * 2));
    document.getElementById('btn-zoom-reset').addEventListener('click', () =>
        fitToRecs(currentRecs));
}

// ============================================================
// Event list (below timeline)
// ============================================================
function renderEventList(recs) {
    elEventList.innerHTML = '';
    const allEvents = [];
    recs.forEach(rec => {
        if (rec.status !== 'ok') return;
        rec.events.forEach(ev => allEvents.push({ rec, ev }));
    });
    allEvents.sort((a, b) => a.ev.abs_start.localeCompare(b.ev.abs_start));

    const filtered = allEvents.filter(({ ev }) => eventPassesFilter(ev));
    const hiddenCount = allEvents.length - filtered.length;
    elEventCount.textContent = hiddenCount > 0
        ? `(${filtered.length} shown, ${hiddenCount} filtered)`
        : `(${filtered.length})`;

    if (!filtered.length) {
        const msg = allEvents.length > 0
            ? `All ${allEvents.length} event(s) hidden by current filters.`
            : 'No events detected on this date.';
        elEventList.innerHTML = `<div style="color:var(--text-dim);font-size:13px;padding:12px 0">${msg}</div>`;
        return;
    }

    filtered.forEach(({ rec, ev }, queueIdx) => {
        const meta = EVENT_META[ev.event_type] ?? { label: ev.event_type, cls: '' };
        const card = document.createElement('div');
        card.className = `event-card ${meta.cls}`;
        card.dataset.autoplayIdx = queueIdx;  // used by highlightAutoplayCard
        card.dataset.filePath = rec.file_path;
        card.dataset.evStart = ev.offset_start;
        card.dataset.evEnd = ev.offset_end;

        const absTime = ev.abs_start.slice(11, 19);  // HH:MM:SS
        const dur = (ev.offset_end - ev.offset_start).toFixed(1);
        const conf = (ev.peak_conf * 100).toFixed(0);

        const tooltip = meta.tooltip || ev.event_type;
        card.innerHTML = `
      <div class="event-card-progress"><div class="event-card-progress-fill"></div></div>
      <div class="event-card-time">${absTime}  (${dur}s)</div>
      <div class="event-card-type" title="${tooltip}">${meta.label}</div>
      <div class="event-card-sev sev-${ev.severity}">${ev.severity}</div>
      <div class="event-card-conf">${conf}%</div>
      <button class="btn-export-event" title="Export clip">⬇ Export</button>`;

        card.querySelector('.btn-export-event').addEventListener('click', e => {
            e.stopPropagation();
            openExportModal(rec, ev);
        });
        card.addEventListener('click', () => {
            if (_autoplayIdx >= 0) {
                // Jump autoplay queue to this event
                playAutoplayItem(queueIdx);
            } else {
                openPlayer(rec, ev.offset_start, ev);
            }
        });
        elEventList.appendChild(card);
    });

    // Show / hide "Play all" button depending on whether there are events
    elBtnPlayAll.classList.toggle('hidden', filtered.length === 0);
}

// ============================================================
// Player
// ============================================================
function openPlayer(rec, offsetSecs, ev) {
    _activeRec = rec;
    const videoUrl = `${API}/video?path=${encodeURIComponent(rec.file_path)}`;
    elVideoEl.src = videoUrl;
    elVideoEl.load();

    // Store silent regions so the skip-silence handler can use them
    _playerSilentRegions = (rec.silent_regions || []);

    elVideoEl.addEventListener('loadedmetadata', () => {
        elVideoEl.currentTime = offsetSecs;
        elVideoEl.play();
    }, { once: true });

    // Labels
    const fname = rec.file_path.split('/').pop();
    elPlayerTitle.textContent = fname;

    if (ev) {
        const meta = EVENT_META[ev.event_type] ?? { label: ev.event_type };
        const tooltip = meta.tooltip || ev.event_type;
        elPlayerBadge.textContent = `${meta.label}  ${fmtOffset(ev.offset_start)} – ${fmtOffset(ev.offset_end)}  (${ev.severity})`;
        elPlayerBadge.title = tooltip;
        elPlayerBadge.style.display = '';
    } else {
        elPlayerBadge.textContent = '';
        elPlayerBadge.title = '';
        elPlayerBadge.style.display = 'none';
    }

    elPlayerDrawer.classList.remove('hidden');
    document.body.classList.add('player-open');
}

function closePlayer() {
    elVideoEl.pause();
    elVideoEl.src = '';
    _playerSilentRegions = [];
    _activeRec = null;
    elTlPlayhead.style.visibility = 'hidden';
    elPlayerTimestamp.textContent = '--:-- / --:--';
    elPlayerGlobalTimestamp.textContent = '--:--:--';
    if (_activeEventCard) { _activeEventCard.classList.remove('playing-active'); _activeEventCard = null; }
    elPlayerDrawer.classList.add('hidden');
    document.body.classList.remove('player-open');
}

// ============================================================
// Autoplay
// ============================================================
function bindAutoplay() {
    elBtnPlayAll.addEventListener('click', startAutoplay);
    elBtnAutoplayStop.addEventListener('click', () => { stopAutoplay(); closePlayer(); });
    elBtnAutoplayNext.addEventListener('click', () => advanceAutoplay(+1));
    elBtnAutoplayPrev.addEventListener('click', () => advanceAutoplay(-1));
}

function buildAutoplayQueue() {
    const all = [];
    currentRecs.forEach(rec => {
        if (rec.status !== 'ok') return;
        rec.events.forEach(ev => { if (eventPassesFilter(ev)) all.push({ rec, ev }); });
    });
    all.sort((a, b) => a.ev.abs_start.localeCompare(b.ev.abs_start));
    return all;
}

function startAutoplay() {
    _autoplayQueue = buildAutoplayQueue();
    if (!_autoplayQueue.length) return;
    playAutoplayItem(0);
}

function stopAutoplay() {
    _autoplayIdx = -1;
    _autoplayQueue = [];
    elAutoplayBar.classList.add('hidden');
    elBtnPlayAll.classList.remove('active');
    // Remove active highlight from all cards
    elEventList.querySelectorAll('.event-card.autoplay-active')
        .forEach(c => c.classList.remove('autoplay-active'));
}

function playAutoplayItem(idx) {
    if (idx < 0 || idx >= _autoplayQueue.length) { stopAutoplay(); closePlayer(); return; }
    _autoplayIdx = idx;
    const { rec, ev } = _autoplayQueue[idx];

    // If the same recording file is already loaded, just seek — avoids reload flicker
    if (_activeRec && _activeRec.file_path === rec.file_path) {
        _activeRec = rec;
        _playerSilentRegions = rec.silent_regions || [];
        elVideoEl.currentTime = ev.offset_start;
        elVideoEl.play();
        const meta = EVENT_META[ev.event_type] ?? { label: ev.event_type };
        elPlayerBadge.textContent = `${meta.label}  ${fmtOffset(ev.offset_start)} – ${fmtOffset(ev.offset_end)}  (${ev.severity})`;
        elPlayerBadge.style.display = '';
    } else {
        openPlayer(rec, ev.offset_start, ev);
    }

    updateAutoplayUI();
    highlightAutoplayCard(idx);
}

function advanceAutoplay(delta) {
    let next = _autoplayIdx + delta;

    // When going forward, skip any events on the same recording that start
    // before the current playback position (would require a backward seek).
    // Also skip events fully inside the current event's play window.
    if (delta > 0) {
        const currentFile = _activeRec?.file_path ?? null;
        const currentEv = _autoplayQueue[_autoplayIdx]?.ev;
        const playWindowEnd = currentEv
            ? currentEv.offset_end + _autoplayPostBuffer
            : elVideoEl.currentTime;
        while (
            next < _autoplayQueue.length &&
            _autoplayQueue[next].rec.file_path === currentFile &&
            _autoplayQueue[next].ev.offset_start < playWindowEnd
        ) {
            next++;
        }
    }

    playAutoplayItem(next);
}

function updateAutoplayUI() {
    const total = _autoplayQueue.length;
    const current = _autoplayIdx + 1;
    elAutoplayCounter.textContent = `Event ${current} / ${total}`;
    elAutoplayBar.classList.remove('hidden');
    elBtnPlayAll.classList.add('active');
    elBtnAutoplayPrev.disabled = _autoplayIdx === 0;
    elBtnAutoplayNext.disabled = _autoplayIdx === total - 1;
}

function highlightAutoplayCard(idx) {
    elEventList.querySelectorAll('.event-card.autoplay-active')
        .forEach(c => c.classList.remove('autoplay-active'));
    const card = elEventList.querySelector(`.event-card[data-autoplay-idx="${idx}"]`);
    if (card) {
        card.classList.add('autoplay-active');
        card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
}

// Skip-silence: jump over silent regions during playback
elVideoEl.addEventListener('timeupdate', () => {
    updatePlayhead();
    updatePlayerTimestamp();
    updateGlobalTimestamp();
    updateEventCardProgress(_activeRec?.file_path, elVideoEl.currentTime);

    // Autoplay: advance when playback passes the current event's end + post-buffer
    if (_autoplayIdx >= 0 && _autoplayQueue[_autoplayIdx]) {
        const { ev } = _autoplayQueue[_autoplayIdx];
        if (elVideoEl.currentTime >= ev.offset_end + _autoplayPostBuffer) {
            advanceAutoplay(+1);
            return;
        }
    }

    if (!_skipSilence || !_playerSilentRegions.length) return;
    const t = elVideoEl.currentTime;
    for (const [s, e] of _playerSilentRegions) {
        // Enter silence zone (with a 0.15 s grace so we don't loop)
        if (t >= s && t < e - 0.15) {
            elVideoEl.currentTime = e;
            return;
        }
    }
});

// If the video reaches its end mid-autoplay (event was near the end of a recording),
// advance to the next event instead of stopping.
elVideoEl.addEventListener('ended', () => {
    if (_autoplayIdx >= 0) advanceAutoplay(+1);
});

// Workaround for a Safari/WebKit bug where seeking drops the audio decoder:
// after every seek, if the video appears un-muted but audio is gone, toggling
// the muted flag forces the browser to reconnect the audio pipeline.
elVideoEl.addEventListener('seeked', () => {
    if (!elVideoEl.muted && elVideoEl.volume > 0) {
        elVideoEl.muted = true;
        elVideoEl.muted = false;
    }
});

// Manual audio reconnect button — for when the automatic seeked workaround
// isn't enough (e.g. fMP4 + Opus on Safari: full media element cycle needed).
elBtnFixAudio.addEventListener('click', () => {
    const src = elVideoEl.src;
    if (!src) return;
    const t = elVideoEl.currentTime;
    const wasPlaying = !elVideoEl.paused;
    const btn = elBtnFixAudio;

    btn.textContent = '⏳ Fixing…';
    btn.disabled = true;

    elVideoEl.pause();
    elVideoEl.src = '';
    elVideoEl.load();
    elVideoEl.src = src;
    elVideoEl.load();
    elVideoEl.addEventListener('loadedmetadata', () => {
        elVideoEl.currentTime = t;
        if (wasPlaying) elVideoEl.play();
        btn.textContent = '🔊 Fix audio';
        btn.disabled = false;
    }, { once: true });
});

// Toggle skip-silence with the button in the player toolbar
elBtnSkipSilence.addEventListener('click', () => {
    _skipSilence = !_skipSilence;
    elBtnSkipSilence.classList.toggle('active', _skipSilence);
    elBtnSkipSilence.title = _skipSilence ? 'Skip silence: ON' : 'Skip silence: OFF';
});

// Center the timeline view on the current playback position
elBtnGotoMarker.addEventListener('click', () => {
    if (!_activeRec) return;
    const absSecs = secsSinceMidnight(_activeRec.rec_start) + elVideoEl.currentTime;
    const span = view.end - view.start;
    setView(absSecs - span / 2, absSecs + span / 2);
});

elBtnAutoFollow.addEventListener('click', () => {
    _autoFollow = !_autoFollow;
    elBtnAutoFollow.classList.toggle('active', _autoFollow);
    elBtnAutoFollow.title = _autoFollow
        ? 'Auto-follow: keep timeline centred on playback position'
        : 'Auto-follow: OFF';
});

// Playhead position — absolute seconds-since-midnight mapped into the view window
function updatePlayhead() {
    if (!_activeRec) {
        elTlPlayhead.style.visibility = 'hidden';
        updateEventCardProgress(null, 0);
        return;
    }
    const absSecs = secsSinceMidnight(_activeRec.rec_start) + elVideoEl.currentTime;
    const pct = (absSecs - view.start) / (view.end - view.start) * 100;

    // Auto-follow: pan the view so the playhead stays near the centre
    if (_autoFollow && !_inAutoFollow) {
        _inAutoFollow = true;
        const span = view.end - view.start;
        setView(absSecs - span / 2, absSecs + span / 2);
        _inAutoFollow = false;
    }

    const pctAfterPan = _autoFollow ? 50 : pct;
    if (pctAfterPan < 0 || pctAfterPan > 100) {
        elTlPlayhead.style.visibility = 'hidden';
    } else {
        elTlPlayhead.style.left = pctAfterPan.toFixed(4) + '%';
        elTlPlayhead.style.visibility = 'visible';
    }
    updateEventCardProgress(_activeRec.file_path, elVideoEl.currentTime);
    scrollToActiveEventCard();
}

let _activeEventCard = null;  // currently highlighted event card

function scrollToActiveEventCard() {
    if (!_activeRec) {
        if (_activeEventCard) { _activeEventCard.classList.remove('playing-active'); _activeEventCard = null; }
        return;
    }
    const t = elVideoEl.currentTime;
    let bestCard = null;
    elEventList.querySelectorAll('.event-card').forEach(card => {
        if (card.dataset.filePath !== _activeRec.file_path) return;
        const start = parseFloat(card.dataset.evStart);
        const end = parseFloat(card.dataset.evEnd);
        if (t >= start && t <= end) bestCard = card;
    });

    // Only switch highlight when playback enters a new event window.
    // Between events (bestCard === null) keep the previous card highlighted.
    if (bestCard !== null && bestCard !== _activeEventCard) {
        if (_activeEventCard) _activeEventCard.classList.remove('playing-active');
        _activeEventCard = bestCard;
        _activeEventCard.classList.add('playing-active');
        // Scroll so the active card sits ~30% from top, keeping future events visible
        const paneTop = elEventsPane.getBoundingClientRect().top;
        const cardTop = _activeEventCard.getBoundingClientRect().top;
        const offset = cardTop - paneTop - elEventsPane.clientHeight * 0.3;
        elEventsPane.scrollBy({ top: offset, behavior: 'smooth' });
    }
}

function updatePlayerTimestamp() {
    const cur = elVideoEl.currentTime;
    const dur = elVideoEl.duration || 0;
    const fmtTime = (s) => {
        const mins = Math.floor(s / 60);
        const secs = Math.floor(s % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };
    elPlayerTimestamp.textContent = `${fmtTime(cur)} / ${fmtTime(dur)}`;
}

function updateGlobalTimestamp() {
    if (!_activeRec) {
        elPlayerGlobalTimestamp.textContent = '-- :--:--';
        return;
    }
    // Parse ISO datetime: "2025-01-15T14:32:45.123Z"
    const startTime = new Date(_activeRec.rec_start);
    const elapsedMs = elVideoEl.currentTime * 1000;
    const currentTime = new Date(startTime.getTime() + elapsedMs);

    const h = currentTime.getHours().toString().padStart(2, '0');
    const m = currentTime.getMinutes().toString().padStart(2, '0');
    const s = currentTime.getSeconds().toString().padStart(2, '0');
    elPlayerGlobalTimestamp.textContent = `${h}:${m}:${s}`;
}

function updateEventCardProgress(filePath, currentTime) {
    elEventList.querySelectorAll('.event-card').forEach(card => {
        const fill = card.querySelector('.event-card-progress-fill');
        if (!fill) return;
        if (!filePath || card.dataset.filePath !== filePath) {
            fill.style.height = '0%';
            return;
        }
        const start = parseFloat(card.dataset.evStart);
        const end = parseFloat(card.dataset.evEnd);
        const dur = end - start;
        if (dur <= 0) { fill.style.height = '100%'; return; }
        const pct = Math.min(100, Math.max(0, (currentTime - start) / dur * 100));
        fill.style.height = pct.toFixed(2) + '%';
    });
}

// Block the video element's own native arrow-key seek (fires at target phase)
// so it doesn't combine with our ±10 s handler below.
elVideoEl.addEventListener('keydown', e => {
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') e.preventDefault();
});

// LEFT / RIGHT arrow keys → ±10 s seek when the player is open
document.addEventListener('keydown', e => {
    if (elPlayerDrawer.classList.contains('hidden')) return;
    const tag = document.activeElement?.tagName ?? '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    if (e.key === 'ArrowLeft') {
        e.preventDefault();
        elVideoEl.currentTime = Math.max(0, elVideoEl.currentTime - 10);
    } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        elVideoEl.currentTime = Math.min(elVideoEl.duration || Infinity, elVideoEl.currentTime + 10);
    }
});

// ============================================================
// Processing
// ============================================================
function startProcessing() {
    const source = elInpSource.value.trim();
    if (!source) { alert('Please enter a path to a file or directory.'); return; }

    elBtnProcess.disabled = true;
    elProgressWrap.classList.remove('hidden');
    setProgress(0, 1, 'Starting…');

    fetch(`${API}/api/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source, force_reprocess: elChkForce.checked }),
    }).then(r => r.json()).then(() => {
        // Start SSE stream
        if (evtSource) evtSource.close();
        evtSource = new EventSource(`${API}/api/process/status`);
        evtSource.onmessage = e => {
            const data = JSON.parse(e.data);
            if (data.done) {
                const s = data.summary;
                setProgress(s.total, s.total,
                    `Done ✓  processed: ${s.processed}  skipped: ${s.skipped}  errors: ${s.errors}`);
                elBtnProcess.disabled = false;
                evtSource.close();
                // Refresh sidebar + stats
                loadDates();
                loadStats();
                if (currentDate) selectDate(currentDate);
            } else if (data.error) {
                setProgress(0, 1, `Error: ${data.error}`);
                elBtnProcess.disabled = false;
                evtSource.close();
            } else {
                setProgress(data.current, data.total, data.msg);
            }
        };
    }).catch(err => {
        setProgress(0, 1, `Request failed: ${err}`);
        elBtnProcess.disabled = false;
    });
}

function setProgress(current, total, msg) {
    const pct = total > 0 ? Math.round((current / total) * 100) : 0;
    elProgressFill.style.width = pct + '%';
    elProgressMsg.textContent = msg;
}

// ============================================================
// Settings modal
// ============================================================
function bindSettings() {
    elBtnSettings.addEventListener('click', () => elSettingsModal.classList.remove('hidden'));
    elBtnCloseSettings.addEventListener('click', () => elSettingsModal.classList.add('hidden'));
    elSettingsModal.addEventListener('click', e => {
        if (e.target === elSettingsModal) elSettingsModal.classList.add('hidden');
    });
    elBtnSaveSettings.addEventListener('click', saveSettings);

    // Load client-side preferences from localStorage
    const saved = localStorage.getItem('autoplayPostBuffer');
    if (saved !== null) _autoplayPostBuffer = parseFloat(saved);
    if (elInpAutoplayBuffer) {
        elInpAutoplayBuffer.value = _autoplayPostBuffer;
        elInpAutoplayBuffer.addEventListener('input', () => {
            _autoplayPostBuffer = Math.max(0, parseFloat(elInpAutoplayBuffer.value) || 0);
        });
    }

    // Live range labels
    const rangeMap = [
        ['thr-cry', 'lbl-cry'],
        ['thr-yell', 'lbl-yell'],
        ['thr-noise', 'lbl-noise'],
        ['thr-talk', 'lbl-talk'],
        ['thr-db', 'lbl-db'],
        ['thr-minsilence', 'lbl-minsilence'],
    ];
    rangeMap.forEach(([rid, lid]) => {
        const inp = document.getElementById(rid);
        const lbl = document.getElementById(lid);
        inp.addEventListener('input', () => { lbl.textContent = inp.value; });
    });
}

async function loadSettingsForm() {
    try {
        const s = await fetchJSON('/api/settings');
        // Weekdays
        document.querySelectorAll('.day-checkboxes input[type=checkbox]').forEach(cb => {
            cb.checked = s.work_days.includes(parseInt(cb.dataset.day));
        });
        document.getElementById('inp-hours-start').value = s.hours_start;
        document.getElementById('inp-hours-end').value = s.hours_end;
        setRange('thr-cry', 'lbl-cry', s.cry_threshold);
        setRange('thr-yell', 'lbl-yell', s.yell_threshold);
        setRange('thr-noise', 'lbl-noise', s.noise_threshold);
        setRange('thr-talk', 'lbl-talk', s.talk_threshold ?? 0.40);
        setRange('thr-db', 'lbl-db', s.silence_db);
        setRange('thr-minsilence', 'lbl-minsilence', s.min_silence_dur);
    } catch { /* default values already in HTML */ }
}

function setRange(rid, lid, val) {
    const inp = document.getElementById(rid);
    const lbl = document.getElementById(lid);
    inp.value = val;
    lbl.textContent = val;
}

async function saveSettings() {
    // Save client-side preferences
    if (elInpAutoplayBuffer) {
        localStorage.setItem('autoplayPostBuffer', elInpAutoplayBuffer.value);
    }

    const days = Array.from(
        document.querySelectorAll('.day-checkboxes input:checked')
    ).map(cb => parseInt(cb.dataset.day));

    const body = {
        work_days: days,
        hours_start: document.getElementById('inp-hours-start').value,
        hours_end: document.getElementById('inp-hours-end').value,
        cry_threshold: parseFloat(document.getElementById('thr-cry').value),
        yell_threshold: parseFloat(document.getElementById('thr-yell').value),
        noise_threshold: parseFloat(document.getElementById('thr-noise').value),
        talk_threshold: parseFloat(document.getElementById('thr-talk').value),
        silence_db: parseFloat(document.getElementById('thr-db').value),
        min_silence_dur: parseFloat(document.getElementById('thr-minsilence').value),
    };

    await fetch(`${API}/api/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });

    elSettingsModal.classList.add('hidden');
}

// ============================================================
// Utilities
// ============================================================
async function fetchJSON(url, opts) {
    const r = await fetch(url, opts);
    if (!r.ok) {
        let msg = `HTTP ${r.status}`;
        try { const d = await r.json(); msg = d.detail || msg; } catch { }
        throw new Error(msg);
    }
    return r.json();
}

function fmtOffset(secs) {
    const m = Math.floor(secs / 60);
    const s = (secs % 60).toFixed(0).padStart(2, '0');
    return `${String(m).padStart(2, '0')}:${s}`;
}

function escHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/"/g, '&quot;');
}
