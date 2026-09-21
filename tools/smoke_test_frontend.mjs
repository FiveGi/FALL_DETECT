/**
 * Load every page of the web app in a real browser and report what is actually rendered.
 *
 * The API smoke test (training/smoke_test_api.py) proves the backend answers; this proves the
 * browser can use those answers. It catches the failures that only appear in a real page --
 * a view that renders blank, a request the page makes that 404s, a JavaScript error that
 * silently stops a list from populating -- none of which show up in a curl test.
 *
 * Needs the frontend on :3000, the backend on :8932, and a Chromium/Edge listening for CDP:
 *   msedge --headless=new --remote-debugging-port=9333 --user-data-dir=<tmp> about:blank
 *   node tools/smoke_test_frontend.mjs
 */
// 127.0.0.1, not localhost: on Windows `localhost` resolves to ::1 first and Docker Desktop's
// WSL port relay binds IPv6 separately, so a sulking relay makes every request hang instead of
// failing fast. Override with CDP_URL / APP_URL if the services live elsewhere.
const CDP = process.env.CDP_URL || 'http://127.0.0.1:9333';
const APP = process.env.APP_URL || 'http://127.0.0.1:3000';
const USER = process.env.SMOKE_USER || 'admin';
const PASSWORD = process.env.SMOKE_PASSWORD || 'admin123';

// The alert list is the one page with content worth asserting on, so it gets specific
// expectations; the rest are checked for "renders, no errors".
const PAGES = [
  ['/dashboard', 'Dashboard'],
  ['/monitor', 'Monitor'],
  ['/camera', 'Cameras'],
  ['/users', 'Users'],
  ['/notification-settings', 'Notification settings'],
  ['/system-logs', 'System logs'],
  ['/thai-frat-list', 'Thai-FRAT'],
  ['/about', 'About'],
];

async function jsonNew(url) {
  const r = await fetch(`${CDP}/json/new?${encodeURIComponent(url)}`, { method: 'PUT' });
  return r.json();
}
function connect(u) {
  return new Promise((res, rej) => {
    const ws = new WebSocket(u);
    ws.addEventListener('open', () => res(ws));
    ws.addEventListener('error', rej);
  });
}
let id = 1;
const send = (ws, method, params = {}) => new Promise(res => {
  const i = id++;
  const h = e => { const d = JSON.parse(e.data); if (d.id === i) { ws.removeEventListener('message', h); res(d.result); } };
  ws.addEventListener('message', h);
  ws.send(JSON.stringify({ id: i, method, params }));
});
const sleep = ms => new Promise(r => setTimeout(r, ms));
const ev = async (ws, expr) => JSON.parse((await send(ws, 'Runtime.evaluate', { expression: expr, returnByValue: true })).result.value);
// For expressions that have to wait for the page -- opening a dialog, letting a v-if render --
// so the check sees the same DOM a person would rather than the one that existed 0ms after a
// click. Returns the resolved value directly; `ev` above would JSON.parse a Promise into {}.
const evAsync = async (ws, expr) => (await send(ws, 'Runtime.evaluate',
  { expression: expr, returnByValue: true, awaitPromise: true })).result.value;

const failures = [];

async function main() {
  const tab = await jsonNew(APP + '/');
  const ws = await connect(tab.webSocketDebuggerUrl);
  let jsErrors = [], httpErrors = [];
  ws.addEventListener('message', e => {
    const d = JSON.parse(e.data);
    if (d.method === 'Runtime.exceptionThrown') jsErrors.push(d.params.exceptionDetails.text.slice(0, 120));
    if (d.method === 'Runtime.consoleAPICalled' && d.params.type === 'error')
      jsErrors.push(d.params.args.map(a => a.value ?? a.description ?? '').join(' ').slice(0, 120));
    if (d.method === 'Network.responseReceived' && d.params.response.status >= 400)
      httpErrors.push(`${d.params.response.status} ${d.params.response.url.split('/api/')[1] || d.params.response.url}`);
  });
  await send(ws, 'Runtime.enable'); await send(ws, 'Page.enable'); await send(ws, 'Network.enable');
  await sleep(3500);

  await send(ws, 'Runtime.evaluate', { expression: `(()=>{const i=document.querySelectorAll('input');if(i.length>=2){i[0].value=${JSON.stringify(USER)};i[0].dispatchEvent(new Event('input',{bubbles:true}));i[1].value=${JSON.stringify(PASSWORD)};i[1].dispatchEvent(new Event('input',{bubbles:true}));}const b=[...document.querySelectorAll('button')].find(b=>b.textContent.includes('เข้าสู่ระบบ'));if(b)b.click();})()` });
  await sleep(6000);
  const loggedIn = await ev(ws, `JSON.stringify({ok: location.pathname !== '/'})`);
  if (!loggedIn.ok) { console.log('FAIL  login'); failures.push('login'); }
  else console.log('PASS  login');

  for (const [path, label] of PAGES) {
    jsErrors = []; httpErrors = [];
    await send(ws, 'Page.navigate', { url: APP + path });
    await sleep(7000);
    const o = await ev(ws, `JSON.stringify({
      path: location.pathname,
      textLen: (document.body.innerText||'').trim().length,
      denied: (document.body.innerText||'').includes('ไม่มีสิทธิ์')
    })`);
    const blank = o.textLen < 60;
    const ok = !blank && !o.denied && httpErrors.length === 0 && jsErrors.length === 0;
    console.log(`${ok ? 'PASS' : 'FAIL'}  ${label.padEnd(22)} ${path.padEnd(24)} text=${o.textLen}` +
      (httpErrors.length ? ` http=${[...new Set(httpErrors)].slice(0, 2).join(',')}` : '') +
      (jsErrors.length ? ` js=${[...new Set(jsErrors)].slice(0, 1)}` : '') +
      (blank ? ' BLANK' : '') + (o.denied ? ' ACCESS-DENIED' : ''));
    if (!ok) failures.push(label);
  }

  // The alert list is the product's main surface: assert its parts are actually there.
  await send(ws, 'Page.navigate', { url: APP + '/monitor' });
  await sleep(9000);
  const m = await ev(ws, `JSON.stringify({
    entries: document.querySelectorAll('.log-entry').length,
    tier: document.querySelectorAll('.tier-badge').length,
    ack: document.querySelectorAll('.btn-ack').length,
    clips: document.querySelectorAll('video.alert-clip').length
  })`);
  const alertsOk = m.entries > 0 && m.tier > 0;
  console.log(`${alertsOk ? 'PASS' : 'FAIL'}  alert list                              ` +
    `entries=${m.entries} tier=${m.tier} ack=${m.ack} clips=${m.clips}`);
  if (!alertsOk) failures.push('alert list');

  // Unreadable controls, measured rather than eyeballed. main.css gives .btn-secondary white
  // text for its dark background; a view that overrides the background without the colour
  // leaves white on near-white, and the system log's pager rendered as two empty grey boxes.
  // Only buttons and links are checked, and only ones with visible text, so this stays a
  // contrast check rather than a design opinion.
  const unreadable = [];
  for (const [path, label] of PAGES) {
    await send(ws, 'Page.navigate', { url: APP + path });
    // Wait for the page to settle rather than a flat delay: the system log's pager only
    // exists once its rows have loaded, and a 2.5s wait checked a page with no buttons on it
    // -- the check passed against a bug that was demonstrably on screen.
    let settled = 0;
    for (let i = 0; i < 16; i++) {
      await sleep(500);
      const n = await ev(ws, `document.querySelectorAll('button, a.btn, .btn').length`);
      if (n > 0 && n === settled) break;
      settled = n;
    }
    const found = await ev(ws, `JSON.stringify((() => {
      const lum = (c) => {
        // Parsed without a regex on purpose. This whole expression lives inside a template
        // literal, where a lone backslash is consumed before the string reaches the browser:
        // /[\d.]+/ silently became /[d.]+/, matched nothing in "rgb(255, 255, 255)", and made
        // this check pass against a bug that was plainly on screen.
        const m = c.replace('rgba(', '').replace('rgb(', '').replace(')', '')
          .split(',').map(v => v.trim()).filter(v => v.length);
        if (m.length < 3) return null;
        if (m.length > 3 && parseFloat(m[3]) < 0.3) return null;   // near-transparent
        const [r, g, b] = m.slice(0, 3).map(v => {
          const x = parseFloat(v) / 255;
          return x <= 0.03928 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
        });
        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
      };
      const bgOf = (el) => {
        for (let n = el; n; n = n.parentElement) {
          const c = getComputedStyle(n).backgroundColor;
          const l = lum(c);
          if (l !== null) return l;
        }
        return 1;
      };
      const bad = [];
      for (const el of document.querySelectorAll('button, a.btn, .btn')) {
        const text = (el.textContent || '').trim();
        const r = el.getBoundingClientRect();
        if (!text || r.width < 4 || r.height < 4) continue;
        const cs = getComputedStyle(el);
        if (cs.visibility === 'hidden' || cs.display === 'none') continue;
        // A gradient or image background reports backgroundColor as transparent, so walking
        // up to the parent would compare the text against the page behind the button and
        // call white-on-green white-on-white. Skipped rather than guessed: this check should
        // only report contrast it can actually compute.
        if (cs.backgroundImage && cs.backgroundImage !== 'none') continue;
        const fg = lum(cs.color);
        if (fg === null) continue;
        const bg = bgOf(el);
        const ratio = (Math.max(fg, bg) + 0.05) / (Math.min(fg, bg) + 0.05);
        if (ratio < 2) bad.push(text.slice(0, 18) + ' (' + ratio.toFixed(1) + ':1)');
      }
      return bad.slice(0, 4);
    })())`);
    if (found.length) unreadable.push(`${label}: ${found.join(', ')}`);
  }
  const readable = unreadable.length === 0;
  console.log(`${readable ? 'PASS' : 'FAIL'}  button text is readable                  ` +
    (readable ? `${PAGES.length} pages` : unreadable.join('; ')));
  if (!readable) failures.push('button contrast');

  // The test-clip dropdown, opened the way a person opens it. It is the first thing anyone
  // trying the system without a camera touches, and two things about it were wrong: the clips
  // were listed 1, 10, 11 ... 17, 2, 3, and each option showed nothing but a number. `17.mp4`
  // has no fall in it, so someone picking it sees no alert and concludes the detector is
  // broken -- which is exactly the conclusion this project drew about its own clip for days.
  await send(ws, 'Page.navigate', { url: APP + '/camera' });
  await sleep(2500);
  const dropdown = await evAsync(ws, `(async () => {
    // The add-camera form is a card on the page, not behind a button; the only thing that
    // has to be clicked is the "test clip" source radio, which is what reveals the dropdown.
    const radio = [...document.querySelectorAll('input[type=radio]')]
      .find(r => (r.closest('label') || {}).textContent?.includes('ไฟล์วิดีโอทดสอบ'));
    if (!radio) return { error: 'no test-clip source option on the add-camera form' };
    radio.click();
    await new Promise(r => setTimeout(r, 1200));
    const select = [...document.querySelectorAll('select')]
      .find(s => [...s.options].some(o => o.textContent.includes('.mp4')));
    if (!select) return { error: 'no test-clip dropdown' };
    const opts = [...select.options].map(o => o.textContent.trim())
      .filter(t => t.includes('.mp4'));
    return { count: opts.length, first: opts[0] || '', second: opts[1] || '',
             last: opts[opts.length - 1] || '' };
  })()`);
  let clipsOk = !dropdown.error && dropdown.count > 0;
  let clipDetail = dropdown.error || `${dropdown.count} clips`;
  if (clipsOk) {
    // 2.mp4 second, not 10.mp4: the list is in human order.
    if (!/^2\.mp4/.test(dropdown.second)) {
      clipsOk = false; clipDetail = `sorted wrong: second option is "${dropdown.second}"`;
    } else if (!dropdown.last.includes('—')) {
      clipsOk = false; clipDetail = `no description shown: "${dropdown.last}"`;
    } else {
      clipDetail = `${dropdown.count} clips, described, in order`;
    }
  }
  console.log(`${clipsOk ? 'PASS' : 'FAIL'}  test-clip dropdown                      ${clipDetail}`);
  if (!clipsOk) failures.push('test-clip dropdown');

  // Every page at phone width, asserting nothing hangs off the right edge. A caregiver is
  // more likely to open this on a phone than anywhere else, and the failure is invisible from
  // a desktop browser: the dashboard's camera grid had a fixed 360px minimum that pushed it
  // 27px past a 390px screen, and the camera form kept Bootstrap's negative-margin gutter
  // without the container that pads it back.
  await send(ws, 'Emulation.setDeviceMetricsOverride',
    { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  const overflowing = [];
  for (const [path, label] of PAGES) {
    await send(ws, 'Page.navigate', { url: APP + path });
    await sleep(2500);
    const o = await ev(ws, `JSON.stringify((() => {
      const d = document.documentElement;
      const over = d.scrollWidth - d.clientWidth;
      let worst = '';
      if (over > 0) {
        for (const el of document.querySelectorAll('*')) {
          const r = el.getBoundingClientRect();
          if (r.right > d.clientWidth + 1 && r.width > 0) {
            worst = el.tagName.toLowerCase() + '.' + (el.className || '').toString().slice(0, 30);
            break;
          }
        }
      }
      return { over, worst };
    })())`);
    if (o.over > 0) overflowing.push(`${label} +${o.over}px (${o.worst})`);
  }
  const fitsPhone = overflowing.length === 0;
  console.log(`${fitsPhone ? 'PASS' : 'FAIL'}  fits a 390px phone                      ` +
    (fitsPhone ? `${PAGES.length} pages` : overflowing.join('; ')));
  if (!fitsPhone) failures.push('phone width');
  await send(ws, 'Emulation.clearDeviceMetricsOverride');

  console.log(`\n${failures.length ? 'FAILURES: ' + failures.join(', ') : 'all pages OK'}`);
  await send(ws, 'Page.close'); ws.close();
  process.exit(failures.length ? 1 : 0);
}

main().catch(e => { console.error('SCRIPT ERROR', e); process.exit(1); });
