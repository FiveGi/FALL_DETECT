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

  console.log(`\n${failures.length ? 'FAILURES: ' + failures.join(', ') : 'all pages OK'}`);
  await send(ws, 'Page.close'); ws.close();
  process.exit(failures.length ? 1 : 0);
}

main().catch(e => { console.error('SCRIPT ERROR', e); process.exit(1); });
