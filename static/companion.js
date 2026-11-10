/*
 * PrapAI Floating Interview Companion
 * -----------------------------------
 * A small floating panel that shows the current copilot answer. It renders into its own root on <body>
 * (the vanilla equivalent of a React portal) and can pop out into an always-on-top window (Document
 * Picture-in-Picture, Chrome/Edge desktop). It owns no data: the host page passes a getter for its answers
 * and calls update() when they change, so the panel never opens a connection or makes an API call.
 *
 *   const companion = createCompanion({
 *       getAnswers: () => answers,            // [{ id, question, text, done, skipped }]
 *       parse: parseAnswer,                   // text -> { structured, say, points[], detail }
 *       inline: safeInline,                   // sanitised inline markdown -> HTML
 *       markdown: safeMarkdown,               // sanitised block markdown -> HTML
 *       canRegenerate: () => bool,            // live session and no answer in progress
 *       onRegenerate: (answer) => {},
 *       canAnswerNow: () => bool,             // optional
 *       onAnswerNow: () => {},                // optional
 *       canOpen: () => bool,                  // optional: may the Alt+C shortcut open the panel?
 *       onToast: (message) => {},             // optional
 *   });
 *
 * Views: "compact" (answer only, icon buttons) and "expanded" (question, labels, status), each with its own
 * remembered size. "Minimized" collapses either view to its title bar without losing the answer.
 *
 * Keyboard (Alt / Option): C show/hide · X compact/expanded · = and - panel size · , and . text size
 * · [ and ] background opacity · Esc (inside the panel) minimize/restore.
 */
(function () {
    'use strict';

    const STORAGE_KEY = 'prapCompanion.v2';   // sessionStorage: layout lasts for this tab's session
    const MARGIN = 8;                          // keep the panel this far inside the viewport
    const MIN_W = 220, MIN_H = 130;
    const DEFAULT_SIZES = { compact: { w: 300, h: 250 }, expanded: { w: 420, h: 500 } };
    const ALPHAS = [0.35, 0.5, 0.65, 0.8, 0.95];   // background opacity steps (text stays fully opaque)
    const DEFAULT_ALPHA = 0.8;
    const SIZE_STEP = 1.12;
    const FONT_STEPS = [0.85, 1, 1.15, 1.3, 1.5];   // answer text size multipliers
    const PIP_SUPPORTED = 'documentPictureInPicture' in window;
    const IS_MAC = /Mac|iPhone|iPad/i.test(navigator.platform || navigator.userAgent || '');
    const ALT = IS_MAC ? '⌥' : 'Alt+';
    const SHORTCUTS = {   // KeyboardEvent.code (layout-independent) -> action
        KeyC: 'toggle', KeyX: 'view', Equal: 'grow', Minus: 'shrink', BracketLeft: 'fainter', BracketRight: 'stronger',
        Comma: 'textDown', Period: 'textUp',
    };

    const svg = (body, fill) => `<svg viewBox="0 0 24 24" width="15" height="15" fill="${fill ? 'currentColor' : 'none'}" ` +
        `stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;
    const ICONS = {
        prev: svg('<path d="M15 6l-6 6 6 6"/>'),
        next: svg('<path d="M9 6l6 6-6 6"/>'),
        copy: svg('<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/>'),
        regen: svg('<path d="M20 12a8 8 0 1 1-2.3-5.7"/><path d="M20 4v5h-5"/>'),
        answer: svg('<path d="M7 5l11 7-11 7z"/>', true),
        alpha: svg('<circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 0 1 0 16z" fill="currentColor"/>'),
        expand: svg('<path d="M14 4h6v6M10 20H4v-6M20 4l-6 6M4 20l6-6"/>'),
        compact: svg('<path d="M4 14h6v6M20 10h-6V4M10 14l-6 6M14 10l6-6"/>'),
        popOut: svg('<path d="M14 4h6v6M20 4l-8 8"/><path d="M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>'),
        popIn: svg('<path d="M20 4l-8 8M12 6v6h6"/><path d="M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>'),
        min: svg('<path d="M6 12h12"/>'),
        restore: svg('<rect x="5" y="5" width="14" height="14" rx="2"/>'),
        close: svg('<path d="M6 6l12 12M18 6L6 18"/>'),
    };

    function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
    const isFiniteNum = (n) => typeof n === 'number' && Number.isFinite(n);

    function loadState() {
        try {
            const s = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || 'null');
            if (s && typeof s === 'object') return s;
        } catch (e) { /* storage blocked or corrupt: use defaults */ }
        return {};
    }

    function saveState(state) {
        try {
            sessionStorage.setItem(STORAGE_KEY, JSON.stringify({
                x: state.x, y: state.y, sizes: state.sizes, view: state.view,
                minimized: state.minimized, alpha: state.alpha, font: state.font,
            }));
        } catch (e) { }
    }

    // "**word**" / "`code`" -> plain text for the clipboard
    function plain(md) {
        return String(md || '').replace(/\*\*(.+?)\*\*/g, '$1').replace(/__(.+?)__/g, '$1')
            .replace(/(^|[^*])\*([^*]+)\*/g, '$1$2').replace(/`([^`]+)`/g, '$1').trim();
    }

    // `win` is the window the panel lives in (the pop-out window has focus, the main page doesn't)
    async function copyText(text, win) {
        const doc = win.document;
        try {
            if (win.navigator.clipboard && win.isSecureContext) {
                await win.navigator.clipboard.writeText(text);
                return true;
            }
        } catch (e) { /* fall through to the textarea fallback */ }
        const ta = doc.createElement('textarea');
        ta.value = text;
        ta.setAttribute('readonly', '');
        ta.style.cssText = 'position:fixed;top:-1000px;opacity:0';
        doc.body.appendChild(ta);
        ta.select();
        let ok = false;
        try { ok = doc.execCommand('copy'); } catch (e) { }
        ta.remove();
        return ok;
    }

    const TEMPLATE = `
        <div class="pc-head" data-drag>
            <span class="pc-dot" data-dot></span>
            <span class="pc-status" data-status>Ready</span>
            <span class="pc-spacer"></span>
            <button type="button" class="pc-icon pc-text-btn" data-act="text-down"
                aria-label="Smaller text (${ALT},)" title="Smaller text (${ALT},)">A&minus;</button>
            <button type="button" class="pc-icon pc-text-btn" data-act="text-up"
                aria-label="Bigger text (${ALT}.)" title="Bigger text (${ALT}.)">A+</button>
            <button type="button" class="pc-icon" data-act="alpha"></button>
            <button type="button" class="pc-icon" data-act="view"></button>
            <button type="button" class="pc-icon" data-act="pip" hidden></button>
            <button type="button" class="pc-icon" data-act="min"></button>
            <button type="button" class="pc-icon pc-icon-close" data-act="close"
                aria-label="Close companion (${ALT}C)" title="Close (${ALT}C)">${ICONS.close}</button>
        </div>
        <div class="pc-body" data-body>
            <div class="pc-empty" data-empty>Answers appear here the moment PrapAI writes them.</div>
            <div class="pc-answer" data-answer hidden>
                <div class="pc-q" data-q role="button" tabindex="0" title="Show the full question"></div>
                <div class="pc-tag" data-skipped hidden>Skipped</div>
                <div class="pc-say" data-say></div>
                <ul class="pc-points" data-points></ul>
                <div class="pc-explain script-text" data-explain hidden></div>
                <details class="pc-detail" data-detail-wrap hidden>
                    <summary><span class="pc-lbl-long" data-detail-lbl>Code example</span><span class="pc-lbl-short" data-detail-lbl-short>Code</span></summary>
                    <div class="pc-detail-body script-text" data-detail></div>
                </details>
                <details class="pc-detail pc-project" data-project-wrap hidden>
                    <summary><span class="pc-lbl-long">Link it to your project</span><span class="pc-lbl-short">Project</span></summary>
                    <div class="pc-detail-body script-text" data-project></div>
                </details>
                <div class="pc-plain script-text" data-plain hidden></div>
            </div>
        </div>
        <div class="pc-live" data-live hidden>
            <span class="pc-live-label">Hearing</span>
            <span class="pc-live-text" data-live-text></span>
        </div>
        <div class="pc-foot">
            <span class="pc-nav">
                <button type="button" class="pc-icon" data-act="prev" aria-label="Previous answer" title="Previous answer">${ICONS.prev}</button>
                <span class="pc-pos" data-pos></span>
                <button type="button" class="pc-icon" data-act="next" aria-label="Next answer" title="Next answer">${ICONS.next}</button>
            </span>
            <span class="pc-spacer"></span>
            <button type="button" class="pc-btn" data-act="copy" aria-label="Copy answer" title="Copy answer">${ICONS.copy}<span class="pc-lbl" data-copy-lbl>Copy</span></button>
            <button type="button" class="pc-btn" data-act="regen" aria-label="Regenerate answer">${ICONS.regen}<span class="pc-lbl">Regenerate</span></button>
            <button type="button" class="pc-btn pc-btn-primary" data-act="answer-now" aria-label="Answer now" title="Answer what was just heard" hidden>${ICONS.answer}<span class="pc-lbl">Answer</span></button>
        </div>
        <div class="pc-resize" data-resize aria-hidden="true"></div>
    `;

    function createCompanion(opts) {
        const o = Object.assign({
            getAnswers: () => [],
            parse: (t) => ({ structured: false, say: t, points: [], detail: '' }),
            inline: (t) => String(t),
            markdown: (t) => String(t),
            canRegenerate: () => false,
            onRegenerate: () => { },
            canAnswerNow: null,
            onAnswerNow: null,
            canOpen: () => true,
            onToast: null,
        }, opts || {});

        const saved = loadState();
        const sizes = {};
        for (const view of ['compact', 'expanded']) {
            const s = saved.sizes && saved.sizes[view];
            sizes[view] = (s && isFiniteNum(s.w) && isFiniteNum(s.h)) ? { w: s.w, h: s.h } : { ...DEFAULT_SIZES[view] };
        }
        const state = {
            open: false,
            view: saved.view === 'expanded' ? 'expanded' : 'compact',
            minimized: saved.minimized === true,
            alpha: ALPHAS.includes(saved.alpha) ? saved.alpha : DEFAULT_ALPHA,
            font: FONT_STEPS.includes(saved.font) ? saved.font : 1,
            sizes,
            x: isFiniteNum(saved.x) ? saved.x : null,
            y: isFiniteNum(saved.y) ? saved.y : null,
            index: -1,                 // which answer is shown; -1 = follow the latest
            detailOpen: new Map(),     // answer id -> details expanded?
            projectOpen: new Map(),    // answer id -> "link it to your project" expanded?
            qExpanded: false,          // full question shown instead of the short preview
            transcript: '',            // live speech line (what is being heard right now)
            scrollTop: 0,              // answer scroll position, kept across minimize/restore
            statusState: 'idle',
            statusText: 'Ready',
        };

        let root = null, el = {};
        let pipWin = null;           // the pop-out window while the panel lives in it
        let updatePending = false;

        const size = () => state.sizes[state.view];

        // ---------- Mounting (portal root on <body>) ----------
        function mount() {
            if (root) return;
            root = document.createElement('div');
            root.className = 'pc-root';
            root.innerHTML = `<section class="pc-panel" role="dialog" aria-label="Interview companion" tabindex="-1">${TEMPLATE}</section>`;
            document.body.appendChild(root);

            const panel = root.firstElementChild;
            const q = (sel) => panel.querySelector(sel);
            el = {
                panel, head: q('[data-drag]'), dot: q('[data-dot]'), status: q('[data-status]'),
                pos: q('[data-pos]'), body: q('[data-body]'), empty: q('[data-empty]'), answer: q('[data-answer]'),
                q: q('[data-q]'), skipped: q('[data-skipped]'), say: q('[data-say]'), points: q('[data-points]'),
                detailWrap: q('[data-detail-wrap]'), detail: q('[data-detail]'), plain: q('[data-plain]'),
                projectWrap: q('[data-project-wrap]'), project: q('[data-project]'), explain: q('[data-explain]'),
                detailLbl: q('[data-detail-lbl]'), detailLblShort: q('[data-detail-lbl-short]'),
                resize: q('[data-resize]'), live: q('[data-live]'), liveText: q('[data-live-text]'),
                prev: q('[data-act="prev"]'), next: q('[data-act="next"]'), alphaBtn: q('[data-act="alpha"]'),
                viewBtn: q('[data-act="view"]'), pip: q('[data-act="pip"]'), min: q('[data-act="min"]'),
                copy: q('[data-act="copy"]'), copyLbl: q('[data-copy-lbl]'), regen: q('[data-act="regen"]'),
                answerNow: q('[data-act="answer-now"]'),
            };
            el.pip.hidden = !PIP_SUPPORTED;

            const toggleQuestion = () => {
                state.qExpanded = !state.qExpanded;
                el.q.classList.toggle('pc-q-full', state.qExpanded);
                el.q.title = state.qExpanded ? 'Show less' : 'Show the full question';
            };
            el.q.addEventListener('click', toggleQuestion);
            el.q.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleQuestion(); }
            });

            panel.addEventListener('click', onClick);
            panel.addEventListener('keydown', (e) => {
                if (e.key === 'Escape' && !pipWin) { e.stopPropagation(); setMinimized(!state.minimized); }
            });
            el.detailWrap.addEventListener('toggle', () => {
                const a = currentAnswer();
                if (a) state.detailOpen.set(a.id, el.detailWrap.open);
            });
            el.projectWrap.addEventListener('toggle', () => {
                const a = currentAnswer();
                if (a) state.projectOpen.set(a.id, el.projectWrap.open);
            });
            el.head.addEventListener('dblclick', (e) => {
                if (!e.target.closest('button')) setView(state.view === 'compact' ? 'expanded' : 'compact');
            });
            enableDrag();
            enableResize();
            window.addEventListener('resize', () => { if (state.open && !pipWin) applyGeometry(); });
        }

        // ---------- Look: view, minimized, opacity ----------
        function applyLook() {
            const p = el.panel;
            const minimized = state.minimized && !pipWin;
            p.classList.toggle('pc-compact', state.view === 'compact');
            p.classList.toggle('pc-minimized', minimized);
            p.style.setProperty('--pc-alpha', String(state.alpha));
            p.style.setProperty('--pc-font', String(state.font));
            const fi = FONT_STEPS.indexOf(state.font);
            p.querySelector('[data-act="text-down"]').disabled = fi <= 0;
            p.querySelector('[data-act="text-up"]').disabled = fi >= FONT_STEPS.length - 1;

            const pct = Math.round(state.alpha * 100);
            setButton(el.alphaBtn, ICONS.alpha, `Background opacity ${pct}% (${ALT}[ and ${ALT}])`);
            setButton(el.viewBtn, state.view === 'compact' ? ICONS.expand : ICONS.compact,
                `${state.view === 'compact' ? 'Expanded' : 'Compact'} view (${ALT}X)`);
            setButton(el.pip, pipWin ? ICONS.popIn : ICONS.popOut,
                pipWin ? 'Move back into the page' : 'Pop out: stays on top of other tabs and apps');
            setButton(el.min, minimized ? ICONS.restore : ICONS.min, minimized ? 'Restore (Esc)' : 'Minimize (Esc)');
        }

        function setButton(btn, icon, label) {
            if (btn._icon !== icon) { btn.innerHTML = icon; btn._icon = icon; }
            btn.title = label;
            btn.setAttribute('aria-label', label);
        }

        // ---------- Geometry ----------
        function viewport() {
            return { vw: document.documentElement.clientWidth, vh: window.innerHeight };
        }

        // Keep the stored geometry valid for the current viewport (also handles phones / window resizes)
        function fitToViewport() {
            const { vw, vh } = viewport();
            const s = size();
            s.w = clamp(s.w, Math.min(MIN_W, vw - 2 * MARGIN), vw - 2 * MARGIN);
            s.h = clamp(s.h, Math.min(MIN_H, vh - 2 * MARGIN), vh - 2 * MARGIN);
            if (state.x === null || state.y === null) {      // first open: bottom-right corner
                state.x = vw - s.w - 20;
                state.y = vh - s.h - 20;
            }
            const height = state.minimized ? (el.head.offsetHeight || 34) : s.h;
            state.x = clamp(state.x, MARGIN, Math.max(MARGIN, vw - s.w - MARGIN));
            state.y = clamp(state.y, MARGIN, Math.max(MARGIN, vh - height - MARGIN));
        }

        function applyGeometry() {
            applyLook();
            if (pipWin) return;   // the pop-out window fills itself
            fitToViewport();
            const s = size();
            Object.assign(el.panel.style, {
                left: `${state.x}px`, top: `${state.y}px`, width: `${s.w}px`,
                height: state.minimized ? 'auto' : `${s.h}px`,
            });
        }

        function setView(view) {
            if (view === state.view) return;
            state.view = view;
            if (pipWin) resizePip(size().w, size().h);
            applyGeometry();
            saveState(state);
            render();
        }

        function setMinimized(minimized) {
            if (pipWin || minimized === state.minimized) return;
            if (minimized) state.scrollTop = el.body.scrollTop;   // the body is hidden while minimized
            state.minimized = minimized;
            applyGeometry();
            saveState(state);
            if (!minimized) {
                render();
                el.body.scrollTop = state.scrollTop;
            }
        }

        function resizeBy(factor) {
            if (pipWin) {
                resizePip(pipWin.innerWidth * factor, pipWin.innerHeight * factor);
                return;
            }
            if (state.minimized) return;
            const s = size();
            const cx = state.x + s.w / 2, cy = state.y + s.h / 2;   // grow/shrink around the centre
            s.w = Math.round(Math.max(MIN_W, s.w * factor));
            s.h = Math.round(Math.max(MIN_H, s.h * factor));
            state.x = Math.round(cx - s.w / 2);
            state.y = Math.round(cy - s.h / 2);
            applyGeometry();
            saveState(state);
        }

        // The pop-out window can only be resized during a user gesture (a click or key press)
        function resizePip(w, h) {
            try { pipWin.resizeTo(Math.round(Math.max(MIN_W, w)), Math.round(Math.max(MIN_H, h))); } catch (e) { }
        }

        function stepAlpha(dir) {
            const i = ALPHAS.indexOf(state.alpha);
            const next = dir === 0 ? (i + 1) % ALPHAS.length : clamp(i + dir, 0, ALPHAS.length - 1);
            state.alpha = ALPHAS[next];
            applyLook();
            saveState(state);
        }

        function stepFont(dir) {
            const i = clamp(FONT_STEPS.indexOf(state.font) + dir, 0, FONT_STEPS.length - 1);
            state.font = FONT_STEPS[i];
            applyLook();
            saveState(state);
        }

        // Pointer events cover mouse, pen and touch; pointer capture keeps the drag alive outside the panel.
        function trackPointer(handle, { canStart, onMove }) {
            handle.addEventListener('pointerdown', (e) => {
                if (e.button !== 0 || pipWin || !canStart(e)) return;
                e.preventDefault();
                const s = size();
                const start = { px: e.clientX, py: e.clientY, x: state.x, y: state.y, w: s.w, h: s.h };
                try { handle.setPointerCapture(e.pointerId); } catch (err) { /* pointer already gone */ }
                el.panel.classList.add('pc-dragging');
                const move = (ev) => onMove(ev.clientX - start.px, ev.clientY - start.py, start);
                const end = () => {
                    handle.removeEventListener('pointermove', move);
                    handle.removeEventListener('pointerup', end);
                    handle.removeEventListener('pointercancel', end);
                    el.panel.classList.remove('pc-dragging');
                    saveState(state);
                };
                handle.addEventListener('pointermove', move);
                handle.addEventListener('pointerup', end);
                handle.addEventListener('pointercancel', end);
            });
        }

        function enableDrag() {
            trackPointer(el.head, {
                canStart: (e) => !e.target.closest('button'),
                onMove: (dx, dy, s) => {
                    state.x = s.x + dx;
                    state.y = s.y + dy;
                    applyGeometry();
                },
            });
        }

        function enableResize() {
            trackPointer(el.resize, {
                canStart: () => !state.minimized,
                onMove: (dx, dy, s) => {
                    const { vw, vh } = viewport();
                    size().w = clamp(s.w + dx, MIN_W, vw - state.x - MARGIN);
                    size().h = clamp(s.h + dy, MIN_H, vh - state.y - MARGIN);
                    applyGeometry();
                },
            });
        }

        // ---------- Keyboard shortcuts ----------
        // Work while focus is in the page or in the pop-out window. Browsers can't see key presses made in
        // other apps, so they don't fire while e.g. the meeting app has focus.
        function onShortcut(e) {
            if (!e.altKey || e.ctrlKey || e.metaKey) return;
            const action = SHORTCUTS[e.code];
            if (!action) return;
            const t = e.target;
            if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName || ''))) return;  // typing
            if (action === 'toggle') {
                if (!state.open && !o.canOpen()) return;
            } else if (!state.open) {
                return;
            }
            e.preventDefault();
            e.stopPropagation();
            switch (action) {
                case 'toggle': state.open ? close() : open(); break;
                case 'view': setView(state.view === 'compact' ? 'expanded' : 'compact'); break;
                case 'grow': resizeBy(SIZE_STEP); break;
                case 'shrink': resizeBy(1 / SIZE_STEP); break;
                case 'fainter': stepAlpha(-1); break;
                case 'stronger': stepAlpha(1); break;
                case 'textDown': stepFont(-1); break;
                case 'textUp': stepFont(1); break;
            }
        }
        window.addEventListener('keydown', onShortcut, true);

        // ---------- Pop-out window (Document Picture-in-Picture) ----------
        // Must run straight from a click: the browser only opens the window during a user gesture.
        async function popOut() {
            if (!PIP_SUPPORTED || pipWin) return;
            let win;
            try {
                win = await window.documentPictureInPicture.requestWindow({
                    width: Math.round(size().w), height: Math.round(size().h),
                });
            } catch (e) {
                if (o.onToast) o.onToast('Could not open the pop-out window.');
                return;
            }
            pipWin = win;
            if (state.minimized) state.scrollTop = 0;
            // Same stylesheets as the page (companion.css + fonts); the app's inline styles aren't needed
            document.querySelectorAll('link[rel="stylesheet"]').forEach((l) => {
                const link = win.document.createElement('link');
                link.rel = 'stylesheet';
                link.href = l.href;
                win.document.head.appendChild(link);
            });
            win.document.title = 'PrapAI Companion';
            win.document.body.className = 'pc-pip-body';
            win.document.body.appendChild(root);   // nodes keep their event listeners when moved
            win.addEventListener('keydown', onShortcut, true);
            el.panel.classList.add('pc-pip');
            el.panel.removeAttribute('style');     // the OS window now handles position and size
            // Closing the pop-out window (its X, or close()) brings the panel back into the page
            win.addEventListener('pagehide', () => {
                pipWin = null;
                updatePending = false;            // timers queued in the closed window never fire
                el.panel.classList.remove('pc-pip');
                document.body.appendChild(root);
                if (state.open) { applyGeometry(); render(); }
            }, { once: true });
            applyLook();
            render();
        }

        // Throttled re-render for streaming text. Timers run in the window the panel is in: a background
        // tab's timers are slowed to once a second, but the visible pop-out window's are not.
        function scheduleUpdate() {
            if (updatePending || !state.open) return;
            updatePending = true;
            (pipWin || window).setTimeout(() => { updatePending = false; render(); }, 60);
        }

        // ---------- Answers ----------
        function answersList() { return o.getAnswers() || []; }

        function currentIndex() {
            const n = answersList().length;
            if (!n) return -1;
            return state.index < 0 || state.index >= n ? n - 1 : state.index;
        }

        function currentAnswer() { return answersList()[currentIndex()] || null; }

        function show(index) {
            const n = answersList().length;
            if (index < 0 || index >= n) return;
            state.index = index === n - 1 ? -1 : index;   // the newest answer means "keep following"
            state.qExpanded = false;
            el.q.classList.remove('pc-q-full');
            render();
            el.body.scrollTop = 0;
        }

        function copyableText(a) {
            const p = o.parse(a.text);
            if (!p.structured) return plain(a.text);
            const parts = [];
            if (p.say) parts.push(plain(p.say));
            if (p.points.length) parts.push(p.points.map(pt => `• ${plain(pt)}`).join('\n'));
            if (p.explain) parts.push(plain(p.explain));
            if (p.code || p.detail) parts.push(p.code || p.detail);
            if (p.project) parts.push(`Link to my project: ${plain(p.project)}`);
            return parts.join('\n\n');
        }

        function render() {
            if (!root || !state.open) return;
            const list = answersList();
            const idx = currentIndex();
            const a = list[idx];

            el.pos.textContent = list.length > 1 ? `${idx + 1}/${list.length}` : '';
            el.prev.disabled = idx <= 0;
            el.next.disabled = idx < 0 || idx >= list.length - 1;
            el.empty.hidden = !!a;
            el.answer.hidden = !a;
            el.copy.disabled = !a || !a.text.trim();
            el.regen.disabled = !a || !a.done || !o.canRegenerate();
            el.regen.title = el.regen.disabled ? 'Regenerate (available in a live session once the answer is complete)' : 'Regenerate: write a fresh version';
            el.answerNow.hidden = !(typeof o.onAnswerNow === 'function' && o.canAnswerNow && o.canAnswerNow());
            el.panel.classList.toggle('pc-streaming', !!a && !a.done);
            if (!a || (state.minimized && !pipWin)) return;   // the answer is kept and drawn on restore

            el.q.textContent = a.question || '';
            el.skipped.hidden = !a.skipped;

            const p = o.parse(a.text);
            // Same rule as the main screen: long unlabelled replies read best as plain markdown
            const usePlain = a.done && !p.structured &&
                (a.text.length > 300 || /\n\s*\n|```|^\s*(?:[-*•]|\d+[.)])\s/m.test(a.text));
            el.plain.hidden = !usePlain;
            el.say.hidden = usePlain;
            el.points.hidden = usePlain || !p.points.length;

            if (usePlain) {
                el.plain.innerHTML = o.markdown(a.text);
                el.detailWrap.hidden = true;
                el.projectWrap.hidden = true;
                el.explain.hidden = true;
                return;
            }
            // The plain-words explanation is always visible
            el.explain.hidden = !p.explain;
            if (p.explain) el.explain.innerHTML = o.markdown(p.explain);
            // First sentence prominent, the explanation after it smaller (same as the main screen)
            el.say.innerHTML = p.sayRest
                ? `${o.inline(p.sayLead)}<span class="pc-say-rest">${o.inline(p.sayRest)}</span>`
                : o.inline(p.say);
            el.points.innerHTML = p.points.map(pt => `<li>${o.inline(pt)}</li>`).join('');
            el.projectWrap.hidden = !p.project;
            if (p.project) {
                el.project.innerHTML = o.markdown(p.project);
                const wantProject = !!state.projectOpen.get(a.id);
                if (el.projectWrap.open !== wantProject) el.projectWrap.open = wantProject;
            }
            // Collapsible: "Code example" (new format) or "More detail" (older answers)
            const extra = p.code || p.detail;
            el.detailWrap.hidden = !extra;
            if (extra) {
                el.detailLbl.textContent = p.code ? 'Code example' : 'More detail';
                el.detailLblShort.textContent = p.code ? 'Code' : 'More';
                el.detail.innerHTML = o.markdown(p.code ||
                    String(p.detail).replace(/^(\s*)\**Example\**:\**/gm, '$1**Example:**'));
                // Older answers with code open automatically; the code example opens when tapped
                if (!state.detailOpen.has(a.id) && !p.code && p.detail.includes('```')) state.detailOpen.set(a.id, true);
                const want = !!state.detailOpen.get(a.id);
                if (el.detailWrap.open !== want) el.detailWrap.open = want;
            }
        }

        function flashCopied(ok) {
            el.copy.classList.toggle('pc-flash', true);
            el.copyLbl.textContent = ok ? 'Copied' : 'Failed';
            el.copy.title = ok ? 'Copied' : 'Copy failed';
            clearTimeout(el.copy._t);
            el.copy._t = setTimeout(() => {
                el.copy.classList.remove('pc-flash');
                el.copyLbl.textContent = 'Copy';
                el.copy.title = 'Copy answer';
            }, 1500);
        }

        async function onClick(e) {
            const btn = e.target.closest('button[data-act]');
            if (!btn || btn.disabled) return;
            const a = currentAnswer();
            switch (btn.dataset.act) {
                case 'prev': show(currentIndex() - 1); break;
                case 'next': show(currentIndex() + 1); break;
                case 'alpha': stepAlpha(0); break;
                case 'text-down': stepFont(-1); break;
                case 'text-up': stepFont(1); break;
                case 'view': setView(state.view === 'compact' ? 'expanded' : 'compact'); break;
                case 'min': setMinimized(!state.minimized); break;
                case 'close': close(); break;
                case 'pip':
                    if (pipWin) pipWin.close();
                    else popOut();
                    break;
                case 'copy':
                    if (a) flashCopied(await copyText(copyableText(a), pipWin || window));
                    break;
                case 'regen':
                    if (a && a.done && o.canRegenerate()) {
                        btn.disabled = true;          // re-enabled by the next render once the host allows it
                        state.index = -1;             // follow the new answer when it arrives
                        o.onRegenerate(a);
                    }
                    break;
                case 'answer-now':
                    if (o.onAnswerNow) o.onAnswerNow();
                    break;
            }
        }

        // ---------- Public API ----------
        function open() {
            mount();
            state.open = true;
            root.hidden = false;
            applyGeometry();
            setStatus(state.statusState, state.statusText);
            setTranscript(state.transcript);   // catch up on speech heard while the panel was closed
            render();
            el.panel.focus({ preventScroll: true });
        }

        function close() {
            state.open = false;
            if (root) root.hidden = true;
            if (pipWin) pipWin.close();   // its pagehide handler moves the (hidden) panel back
        }

        function setStatus(stateName, text) {
            state.statusState = stateName || 'idle';
            state.statusText = text || '';
            if (!root) return;
            el.dot.className = `pc-dot ${state.statusState}`;
            el.status.textContent = state.statusText;
            render();   // buttons depend on session state
        }

        function reset() {
            state.index = -1;
            state.qExpanded = false;
            state.scrollTop = 0;
            if (el.q) el.q.classList.remove('pc-q-full');
            state.detailOpen.clear();
            state.projectOpen.clear();
            setTranscript('');
            render();
        }

        // Live speech line. Called for every interim result, so it only touches this one element.
        function setTranscript(text) {
            const t = String(text || '').trim();
            state.transcript = t.length > 160 ? '…' + t.slice(-160) : t;
            if (!root) return;
            el.liveText.textContent = state.transcript;
            el.live.hidden = !state.transcript;
        }

        return {
            open, close, reset, setStatus, setTranscript,
            toggle() { state.open ? close() : open(); },
            isOpen() { return state.open; },
            isPoppedOut() { return !!pipWin; },
            update: render,
            scheduleUpdate,
            shortcutHint: `${ALT}C`,
        };
    }

    window.createCompanion = createCompanion;
})();
