/**
 * Owner, 8 Oct: "test only through what the user sees" -- a clip test driven entirely through the web UI, the way a
 * caregiver would do it, in a real browser over CDP. No API shortcuts:
 *   login -> Cameras: name, room, owner, source "ไฟล์วิดีโอทดสอบ", pick the clip, type "ตรวจจับการล้ม" -> save
 *   -> Monitor: "เริ่มการตรวจจับ" on that camera -> wait for its fall alert in the alert list -> screenshot
 *   -> the alert's clip (60 s before the event) loads and has a duration -> ACK=web: press "รับทราบ" and see
 *      "✓ รับทราบแล้ว" | ACK=line: wait (up to ACK_WAIT s) for the acknowledgement made in LINE to show up after a reload
 *   -> "หยุดการตรวจจับ" -> Cameras: "ลบ". Screenshots go to OUT_DIR.
 * Needs the frontend on :3000, backend :8932, a Chromium/Edge with CDP on :9333 (see smoke_test_frontend.mjs).
 * Usage: CLIP=14 ACK=web node tools/ui_clip_test.mjs
 *        CLIP=17 EXPECT=none WINDOW=180 node tools/ui_clip_test.mjs   (a clip with no fall: PASS = no fall alert in WINDOW s)
 * Every run prints 'START_AT <ISO>' (the moment "เริ่มการตรวจจับ" was pressed) and, for an alert, 'ALERT_TEXT <text>'
 * (its time as the page shows it), so the delay from the fall in the clip can be worked out (full test, 8 Oct).
 */
const CDP = process.env.CDP_URL || 'http://127.0.0.1:9333';
const APP = process.env.APP_URL || 'http://127.0.0.1:3000';
const CLIP = process.env.CLIP || '14';
const ACK = process.env.ACK || 'web';
const ACK_WAIT = Number(process.env.ACK_WAIT || 300);
const NAME = process.env.NAME || `UI clip${CLIP} ${Date.now() % 100000}`;
const OUT_DIR = process.env.OUT_DIR || '.';
const EXPECT = process.env.EXPECT || 'fall';
const WINDOW = Number(process.env.WINDOW || 180);
const sleep = ms => new Promise(r => setTimeout(r, ms));
import fs from 'node:fs';

let ws, TAB_ID, nextId = 1;
const pending = new Map();
const send = (method, params = {}) => new Promise(res => {
  const id = nextId++; pending.set(id, res); ws.send(JSON.stringify({ id, method, params }));
});
const ev = async expr => (await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true })).result?.value;
const waitFor = async (expr, ms = 20000) => { for (let t = 0; t < ms; t += 500) { if (await ev(`!!(${expr})`)) return true; await sleep(500); } return false; };
const shot = async name => {
  const r = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false });
  fs.writeFileSync(`${OUT_DIR}/${name}.png`, Buffer.from(r.data, 'base64'));
};
const results = [];
const step = (ok, what, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}${detail ? '  -- ' + detail : ''}`); return ok; };
// Set a Vue v-model'd input/select the way typing/choosing does (native setter + input/change events).
const setField = (sel, value) => ev(`(() => { const el = document.querySelector(${JSON.stringify(sel)}); if (!el) return false;
  const proto = el.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, ${JSON.stringify(value)});
  el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true })); return true; })()`);
const clickText = (sel, text, scope = 'document') => ev(`(() => { const b = [...${scope}.querySelectorAll(${JSON.stringify(sel)})]
  .find(x => x.textContent.trim().includes(${JSON.stringify(text)}) && !x.disabled); if (!b) return false; b.click(); return true; })()`);
// EXACT matches only (8 Oct: a loose 'contains the name' selector matched a container of several cards and pressed
// another camera's stop/delete button). A card / list row is ONE camera whose name element equals NAME.
const card = `[...document.querySelectorAll('.camera-card')].find(c => [...c.querySelectorAll('h1,h2,h3,h4,.camera-name,.camera-title')].some(h => h.textContent.trim() === ${JSON.stringify(NAME)}))`;
const row = `[...document.querySelectorAll('.camera-item')].find(r => r.querySelector('.camera-name')?.textContent.trim() === ${JSON.stringify(NAME)})`;
// The ALERT entry (with the acknowledge button): a notification ('การแจ้งเตือน') about a fall, for this camera -- not a
// detection-log line like 'ไม่ตรวจพบการล้ม' (which also contains 'ล้ม').
const entry = `[...document.querySelectorAll('.log-entry')].find(e => e.innerText.includes(${JSON.stringify(NAME)})
  && e.innerText.includes('การแจ้งเตือน') && /ล้ม/.test(e.innerText) && !e.innerText.includes('ไม่ตรวจพบ'))`;

async function main() {
  const tab = await (await fetch(`${CDP}/json/new?${APP}/`, { method: 'PUT' })).json();
  TAB_ID = tab.id;
  ws = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise(r => ws.addEventListener('open', r));
  ws.addEventListener('message', e => { const d = JSON.parse(e.data); if (d.id && pending.has(d.id)) { pending.get(d.id)(d.result || {}); pending.delete(d.id); } });
  await send('Runtime.enable'); await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: 1366, height: 900, deviceScaleFactor: 1, mobile: false });
  await sleep(3000);
  await ev(`(() => { const i = document.querySelectorAll('input'); const s = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
    s.call(i[0], 'admin'); i[0].dispatchEvent(new Event('input', { bubbles: true })); s.call(i[1], 'admin123'); i[1].dispatchEvent(new Event('input', { bubbles: true }));
    [...document.querySelectorAll('button')].find(b => b.textContent.includes('เข้าสู่ระบบ')).click(); })()`);
  step(await waitFor(`location.pathname !== '/'`), 'login');

  await send('Page.navigate', { url: `${APP}/camera` });
  await waitFor(`document.querySelector('#camera-name')`);
  await setField('#camera-name', NAME); await setField('#room-name', 'ทดสอบผ่านหน้าเว็บ');
  // The owner list loads after the form (8 Oct full test: clip 15 saved nothing because the list was still empty).
  await waitFor(`[...document.querySelectorAll('select')].some(x => [...x.options].some(o => o.value && /admin/i.test(o.textContent)))`, 15000);
  const owner = await ev(`(() => { const s = document.querySelector('#owner-select') || [...document.querySelectorAll('select')].find(x => [...x.options].some(o => /admin/i.test(o.text)));
    if (!s) return ''; const o = [...s.options].find(o => o.value && /admin/i.test(o.textContent)) || [...s.options].find(o => o.value);
    return o ? (s.id ? '#' + s.id : '') + '|' + o.value : ''; })()`);
  if (owner && owner.startsWith('#')) await setField(owner.split('|')[0], owner.split('|')[1]);
  await ev(`[...document.querySelectorAll('input[type=radio]')].find(r => r.value === 'test')?.click()`);
  step(await waitFor(`[...(document.querySelector('#camera-test-video')?.options || [])].some(o => o.textContent.startsWith(${JSON.stringify(CLIP + '.mp4')}))`),
    `clip dropdown lists ${CLIP}.mp4`);
  const url = await ev(`[...document.querySelector('#camera-test-video').options].find(o => o.textContent.startsWith(${JSON.stringify(CLIP + '.mp4')})).value`);
  await setField('#camera-test-video', url);
  await setField('#detection-type', 'fall_v2');
  await shot(`ui_${CLIP}_1_form`);
  await clickText('button', 'บันทึกและไปที่มอนิเตอร์');
  step(await waitFor(`location.pathname === '/monitor'`, 15000), 'saved and moved to the Monitor page');
  step(await waitFor(card, 20000), 'camera card shown on Monitor');
  await ev(`(() => { const c = ${card}; [...c.querySelectorAll('button')].find(b => b.textContent.includes('เริ่มการตรวจจับ')).click(); })()`);
  console.log('START_AT', new Date().toISOString());
  step(await waitFor(`(${card}).innerText.includes('หยุดการตรวจจับ')`, 30000), 'start detection pressed -> camera running');
  // A user looks at THEIR camera: click its card so the alert list shows that camera (the page keeps whichever camera
  // was selected before -- finding of 8 Oct: right after saving, the list still shows another camera's alerts).
  const before = await ev(`document.querySelector('.camera-card.camera-selected .camera-title')?.textContent.trim() || ''`);
  console.log('   camera selected on the Monitor page after saving:', JSON.stringify(before));
  if (before !== NAME) await ev(`(${card}).querySelector('.camera-title').click()`);   // only if not already ours
  step(await waitFor(`document.querySelector('.camera-card.camera-selected .camera-title')?.textContent.trim() === ${JSON.stringify(NAME)}`, 5000),
    'our camera is the selected one (its alerts are listed)');
  const t0 = Date.now();
  let got;
  if (EXPECT === 'none') {
    // Absence needs a stricter watch than waitFor (Codex P2, 8 Oct): look right up to the deadline (a last check after
    // it), and a check that could not run (page error) is NOT "no alert" -- any such error fails the step.
    let errors = 0, polls = 0; got = false;
    const end = t0 + WINDOW * 1000;
    for (;;) {
      const r = await send('Runtime.evaluate', { expression: `!!(${entry})`, returnByValue: true });
      polls++;
      if (r.exceptionDetails || typeof r.result?.value !== 'boolean') errors++;
      else if (r.result.value) { got = true; break; }
      if (Date.now() >= end) break;
      await sleep(Math.min(500, Math.max(0, end - Date.now())));
    }
    step(!got && errors === 0, `no fall alert for a clip with no fall (watched ${WINDOW} s)`,
      got ? 'ALERT after ' + Math.round((Date.now() - t0) / 1000) + ' s' : `${polls} checks, ${errors} could not run`);
  } else {
    got = await waitFor(entry, WINDOW * 1000);
    step(got, 'fall alert appears in the alert list', got ? `after ${Math.round((Date.now() - t0) / 1000)} s` : `none in ${WINDOW} s`);
  }
  if (got) console.log('ALERT_TEXT', JSON.stringify(await ev(`(${entry}).innerText.replace(/\\s+/g, ' ').slice(0, 220)`)));
  if (!got) console.log('   what the list shows:', JSON.stringify(await ev(`({n: document.querySelectorAll('.log-entry').length,
    empty: !!document.querySelector('.empty-log'), first: [...document.querySelectorAll('.log-entry')].slice(0, 4).map(e => e.innerText.replace(/\s+/g, ' ').slice(0, 160))})`)));
  await shot(`ui_${CLIP}_2_alert`);
  if (got && EXPECT !== 'none') {
    // The list is re-rendered every 10 s (polling), so a <video> grabbed from it can be replaced mid-wait. Take the URL
    // the page shows and play it in a detached element the poll cannot replace -- the same URL the user's player loads.
    const clip = await ev(`(async () => { const src = (${entry}).querySelector('video.alert-clip')?.src; if (!src) return 'no clip in the alert';
      const v = document.createElement('video'); v.preload = 'auto'; v.muted = true; v.src = src; v.load();
      for (let i = 0; i < 60 && v.readyState < 1 && !v.error; i++) await new Promise(r => setTimeout(r, 250));
      return v.error ? 'error ' + v.error.code : v.readyState >= 1 ? 'duration ' + v.duration.toFixed(1) + ' s' : 'did not load'; })()`);
    // Length is NOT required to be 60 s: the clip holds what the camera saw since it started (a fall 10 s after start
    // gives a ~5-10 s clip); it must exist and play.
    step(/^duration [\d.]+/.test(clip) && Number(clip.split(' ')[1]) > 1, 'alert clip (up to 60 s before) plays from the page', clip);
    if (ACK === 'web') {
      await ev(`(${entry}).querySelector('.btn-ack')?.click()`);
      step(await waitFor(`(${entry}).innerText.includes('รับทราบแล้ว')`, 15000), 'pressed "รับทราบ" in the web -> "✓ รับทราบแล้ว"');
    } else {
      console.log(`waiting up to ${ACK_WAIT} s for the acknowledgement pressed in LINE ...`);
      let ok = false;
      for (let t = 0; t < ACK_WAIT && !ok; t += 15) { await sleep(15000); await send('Page.reload'); await waitFor('document.querySelector(".log-entry")', 15000); ok = await ev(`!!(${entry}) && (${entry}).innerText.includes('รับทราบแล้ว')`); }
      step(ok, 'acknowledged in LINE -> the web shows "✓ รับทราบแล้ว"');
    }
    await shot(`ui_${CLIP}_3_ack`);
  }
  await waitFor(card, 10000);
  await ev(`(() => { const c = ${card}; if (!c) return false; [...c.querySelectorAll('button')].find(b => b.textContent.includes('หยุดการตรวจจับ'))?.click(); return true; })()`);
  step(await waitFor(`(${card}).innerText.includes('เริ่มการตรวจจับ')`, 30000), 'stop detection pressed -> camera stopped');
  await send('Page.navigate', { url: `${APP}/camera` });
  await waitFor(`document.body.innerText.includes(${JSON.stringify(NAME)})`, 15000);
  await ev(`window.confirm = () => true`);
  await ev(`(() => { const r = ${row}; if (!r) return false; [...r.querySelectorAll('button')].find(b => b.textContent.trim() === 'ลบ').click(); return true; })()`);
  step(await waitFor(`!document.body.innerText.includes(${JSON.stringify(NAME)})`, 15000), 'camera deleted from the Cameras page');
  console.log(results.every(Boolean) ? 'UI CLIP TEST PASS' : 'UI CLIP TEST FAIL');
  // Close this tab: left open, every run's Monitor page keeps polling and streaming, and after ~a dozen the browser
  // stopped loading new videos (8 Oct: a clip that plays fine stuck at readyState 0 until 12 stale tabs were closed).
  ws.close(); await fetch(`${CDP}/json/close/${TAB_ID}`).catch(() => {});
  process.exit(results.every(Boolean) ? 0 : 1);
}
main().catch(e => { console.log('FAIL  crashed: ' + e.message); process.exit(2); });
