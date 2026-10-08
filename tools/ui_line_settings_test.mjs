/**
 * Owner, 8 Oct: users set up their OWN LINE in the web page -- test that the way a new user would, in a real browser:
 *   admin: Users page "+ เพิ่มผู้ใช้ใหม่" -> new ordinary user
 *   new user: log in -> "ตั้งค่าการแจ้งเตือน" -> sees the system bot's token already saved (single-bot model) ->
 *   types their LINE user id -> switch ON -> "บันทึกการตั้งค่า LINE" -> "ทดสอบการส่งข้อความ" (one real push to that id)
 *   -> switch OFF -> save;   admin: deletes the test user.
 * The LINE user id comes from the env var LINE_TEST_USER_ID (never printed). SEND=0 skips the real test push.
 * Needs the frontend on :3000, backend :8932, Chromium/Edge CDP on :9333. Screenshots to OUT_DIR.
 */
import fs from 'node:fs';
const CDP = process.env.CDP_URL || 'http://127.0.0.1:9333';
const APP = process.env.APP_URL || 'http://127.0.0.1:3000';
const LINE_ID = process.env.LINE_TEST_USER_ID || '';
const SEND = process.env.SEND !== '0';
const OUT_DIR = process.env.OUT_DIR || '.';
const USER = 'linetest' + (Date.now() % 100000), PASS = 'LineTest-' + (Date.now() % 1000000);
const sleep = ms => new Promise(r => setTimeout(r, ms));
let ws, TAB, id = 1; const pending = new Map();
const send = (m, p = {}) => new Promise(r => { const i = id++; pending.set(i, r); ws.send(JSON.stringify({ id: i, method: m, params: p })); });
const ev = async x => (await send('Runtime.evaluate', { expression: x, returnByValue: true, awaitPromise: true })).result?.value;
const waitFor = async (x, ms = 20000) => { for (let t = 0; t < ms; t += 500) { if (await ev(`!!(${x})`)) return true; await sleep(500); } return false; };
const results = []; const step = (ok, what, d = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}${d ? '  -- ' + d : ''}`); return ok; };
const setVal = (sel, v) => ev(`(() => { const el = document.querySelector(${JSON.stringify(sel)}); if (!el) return false;
  const proto = el.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, ${JSON.stringify(v)});
  el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true })); return true; })()`);
const click = text => ev(`(() => { const b = [...document.querySelectorAll('button')].find(b => b.textContent.trim().includes(${JSON.stringify(text)}) && !b.disabled); if (!b) return false; b.click(); return true; })()`);
const shot = async n => { const r = await send('Page.captureScreenshot', { format: 'png' }); fs.writeFileSync(`${OUT_DIR}/${n}.png`, Buffer.from(r.data, 'base64')); };
async function login(u, p) {
  await ev(`localStorage.clear()`); await send('Page.navigate', { url: APP + '/' }); await waitFor(`document.querySelectorAll('input').length >= 2`);
  await ev(`(() => { const i = document.querySelectorAll('input'); const s = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
    s.call(i[0], ${JSON.stringify(u)}); i[0].dispatchEvent(new Event('input', { bubbles: true })); s.call(i[1], ${JSON.stringify(p)}); i[1].dispatchEvent(new Event('input', { bubbles: true }));
    [...document.querySelectorAll('button')].find(b => b.textContent.includes('เข้าสู่ระบบ')).click(); })()`);
  return waitFor(`location.pathname !== '/' && localStorage.getItem('access_token')`);
}
async function main() {
  const tab = await (await fetch(`${CDP}/json/new?${APP}/`, { method: 'PUT' })).json(); TAB = tab.id;
  ws = new WebSocket(tab.webSocketDebuggerUrl); await new Promise(r => ws.addEventListener('open', r));
  ws.addEventListener('message', e => { const d = JSON.parse(e.data); if (d.id && pending.has(d.id)) { pending.get(d.id)(d.result || {}); pending.delete(d.id); } });
  await send('Runtime.enable'); await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: 1366, height: 900, deviceScaleFactor: 1, mobile: false });
  await sleep(2500);
  step(await login('admin', 'admin123'), 'admin logs in');
  await send('Page.navigate', { url: APP + '/users' }); await waitFor(`[...document.querySelectorAll('button')].some(b => b.textContent.includes('เพิ่มผู้ใช้ใหม่'))`);
  await click('เพิ่มผู้ใช้ใหม่'); await waitFor(`document.querySelector('#new-username')`);
  await setVal('#new-username', USER); await setVal('#new-password', PASS); await setVal('#new-role', 'user');
  await ev(`document.querySelector('#new-username').closest('form').requestSubmit()`);
  step(await waitFor(`document.body.innerText.includes(${JSON.stringify(USER)}) && !document.querySelector('#new-username')`, 15000), 'admin creates a new ordinary user in the Users page');

  step(await login(USER, PASS), 'the new user logs in');
  // Like a user: find the menu item and click it (no typed URL).
  step(await waitFor(`[...document.querySelectorAll('a')].some(a => a.textContent.trim() === 'ตั้งค่าการแจ้งเตือน')`, 15000), 'the menu item is visible to an ordinary user');
  await ev(`[...document.querySelectorAll('a')].find(a => a.textContent.trim() === 'ตั้งค่าการแจ้งเตือน').click()`);
  step(await waitFor(`location.pathname === '/notification-settings' && document.querySelector('#line-user-id')`, 20000), 'clicking it opens "ตั้งค่าการแจ้งเตือน" with the LINE id field');
  console.log('   state after click:', JSON.stringify(await ev(`({path: location.pathname, field: !!document.querySelector('#line-user-id'), h1: document.querySelector('h1')?.textContent.trim()})`)));
  await sleep(1500);
  const seeded = await ev(`!document.querySelector('#line-token') && document.body.innerText.includes('ระบบใช้บอท LINE ของระบบอยู่แล้ว')`);
  step(seeded, 'an ordinary user is not asked for the bot token -- only their own LINE id (single-bot model)');
  const ack = await ev(`(document.body.innerText.match(/(พร้อมใช้งาน|ยังใช้ไม่ได้)[^\\n]{0,40}/) || [''])[0]`);
  console.log('   acknowledge-button status shown to the user:', JSON.stringify(ack));
  await shot('ui_line_1_settings');
  if (!LINE_ID) { step(false, 'LINE_TEST_USER_ID not given'); }
  else {
    await setVal('#line-user-id', LINE_ID);
    await ev(`(() => { const c = document.querySelector('.toggle-row input[type=checkbox]'); if (c && !c.checked) c.click(); return !!c; })()`);
    await click('บันทึกการตั้งค่า LINE');
    await sleep(2500); await send('Page.reload');
    await waitFor(`document.querySelector('#line-user-id')?.value`, 20000);
    step(await ev(`document.querySelector('#line-user-id').value === ${JSON.stringify(LINE_ID)} && document.querySelector('.toggle-row input[type=checkbox]').checked`),
      'typed own LINE id, switched ON, saved -- still there after reloading the page');
    if (SEND) {
      await waitFor(`[...document.querySelectorAll('button')].some(b => b.textContent.includes('ทดสอบการส่งข้อความ') && !b.disabled)`, 10000);
      await click('ทดสอบการส่งข้อความ');
      step(await waitFor(`/ส่ง.*(สำเร็จ|แล้ว)|sent to 1/i.test(document.body.innerText)`, 15000), 'pressed "ทดสอบการส่งข้อความ" -> the page reports it sent (check the phone)');
      await shot('ui_line_2_tested');
    }
    await ev(`(() => { const c = document.querySelector('.toggle-row input[type=checkbox]'); if (c && c.checked) c.click(); })()`);
    await click('บันทึกการตั้งค่า LINE'); await sleep(2000);
    step(await ev(`!document.querySelector('.toggle-row input[type=checkbox]').checked`), 'switched LINE OFF again and saved');
  }
  step(await login('admin', 'admin123'), 'admin logs back in');
  await send('Page.navigate', { url: APP + '/users' }); await waitFor(`document.body.innerText.includes(${JSON.stringify(USER)})`);
  await ev(`window.confirm = () => true`);
  await ev(`(() => { const card = [...document.querySelectorAll('.user-item')].find(c => c.querySelector('.user-name')?.textContent.trim() === ${JSON.stringify(USER)});
    [...card.querySelectorAll('button')].find(b => b.textContent.trim() === 'ลบ').click(); })()`);
  step(await waitFor(`![...document.querySelectorAll('.user-item .user-name')].some(n => n.textContent.trim() === ${JSON.stringify(USER)})`, 15000), 'admin deletes the test user');
  console.log(results.every(Boolean) ? 'UI LINE SETTINGS TEST PASS' : 'UI LINE SETTINGS TEST FAIL');
  ws.close(); await fetch(`${CDP}/json/close/${TAB}`).catch(() => {}); await sleep(300);
  process.exitCode = results.every(Boolean) ? 0 : 1;
}
main().catch(e => { console.log('FAIL  crashed: ' + e.message); process.exitCode = 2; });
