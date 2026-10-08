/**
 * Full test, 8 Oct ("ใช้งานให้หมด"): the Thai-FRAT assessment used the way a nurse would, through the web UI only.
 *   login -> menu "ข้อมูล Thai-FRAT" (/thai-frat-list, whose own form is where a new assessment is made): name, phone, province (the custom dropdown), PDPA, six answers -> save
 *   -> the list shows the person with the right total -> "ดูรายละเอียด" shows the same total and a risk level
 *   -> "แก้ไข": change answer 1 -> "อัปเดตข้อมูล" -> the total changes by exactly that answer's points
 *   -> "ลบ" -> the person is gone from the list.
 * (/thai-frat-form is only reached through "แก้ไข". Opened directly in create mode it does NOT save -- its addForm()
 *  is local-only; found by this test on 8 Oct, nothing in the UI links there. Edit mode is tested below.)
 * Answers chosen: q1 25 + q2 15 + q3 15 + q4 0 + q5 10 + q6 0 = 65; after the edit q1 = 0 -> 40.
 * Needs the frontend on :3000 and a Chromium/Edge with CDP on :9333 (see smoke_test_frontend.mjs).
 * Usage: node tools/ui_thaifrat_test.mjs
 */
const CDP = process.env.CDP_URL || 'http://127.0.0.1:9333';
const APP = process.env.APP_URL || 'http://127.0.0.1:3000';
// Unique per run, and every step after the save works on the record ID this run created -- never on a name alone
// (Codex P1, 8 Oct: a reused name could open someone else's record, and the cleanup would then delete it).
const NAME = `ทดสอบ FRAT ${Date.now()}-${Math.random().toString(16).slice(2, 6)}`;
const sleep = ms => new Promise(r => setTimeout(r, ms));

let ws, TAB_ID, nextId = 1;
const pending = new Map();
const send = (method, params = {}) => new Promise(res => {
  const id = nextId++; pending.set(id, res); ws.send(JSON.stringify({ id, method, params }));
});
const ev = async expr => (await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true })).result?.value;
const waitFor = async (expr, ms = 15000) => { for (let t = 0; t < ms; t += 500) { if (await ev(`!!(${expr})`)) return true; await sleep(500); } return false; };
const results = [];
const step = (ok, what, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}${detail ? '  -- ' + detail : ''}`); return ok; };
const typeInto = (sel, value) => ev(`(() => { const el = document.querySelector(${JSON.stringify(sel)}); if (!el) return false;
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, ${JSON.stringify(value)});
  el.dispatchEvent(new Event('input', { bubbles: true })); return true; })()`);
// Choose the option whose text shows these points, the way a user picks "(25 คะแนน)" -- by index, so Vue's
// v-model.number reads the option's bound number.
const answer = (i, points) => ev(`(() => { const s = document.querySelectorAll('.score-inputs select')[${i}]; if (!s) return false;
  const k = [...s.options].findIndex(o => o.textContent.includes('(' + ${points} + ' คะแนน')); if (k < 0) return false;
  s.selectedIndex = k; s.dispatchEvent(new Event('change', { bubbles: true })); return true; })()`);
const clickText = (sel, text) => ev(`(() => { const b = [...document.querySelectorAll(${JSON.stringify(sel)})]
  .find(x => x.textContent.trim().includes(${JSON.stringify(text)}) && !x.disabled); if (!b) return false; b.click(); return true; })()`);
const row = `[...document.querySelectorAll('tbody tr')].find(r => r.cells[0]?.textContent.trim() === ${JSON.stringify(NAME)})`;
// IDs of the records listed, from each row's "ดูรายละเอียด" link (/thai-frat-detail/<id>).
const listIds = `[...document.querySelectorAll('tbody tr a')].map(a => (a.getAttribute('href') || '').match(/thai-frat-detail\\/(\\d+)$/)?.[1]).filter(Boolean)`;
const finish = async () => {
  console.log(results.every(Boolean) ? 'UI THAI-FRAT TEST PASS' : 'UI THAI-FRAT TEST FAIL');
  ws.close(); await fetch(`${CDP}/json/close/${TAB_ID}`).catch(() => {});
  process.exit(results.every(Boolean) ? 0 : 1);
};
// A step whose failure means the next steps could touch a record that is not ours: stop, edit and delete nothing.
const must = async (ok, what, detail = '') => { if (!step(ok, what, detail)) { console.log('   stopping: nothing further is edited or deleted'); await finish(); } };

async function main() {
  const tab = await (await fetch(`${CDP}/json/new?${APP}/`, { method: 'PUT' })).json();
  TAB_ID = tab.id;
  ws = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise(r => ws.addEventListener('open', r));
  ws.addEventListener('message', e => { const d = JSON.parse(e.data); if (d.id && pending.has(d.id)) { pending.get(d.id)(d.result || {}); pending.delete(d.id); } });
  await send('Runtime.enable'); await send('Page.enable');
  await sleep(3000);
  await ev(`(() => { const i = document.querySelectorAll('input'); const s = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
    s.call(i[0], 'admin'); i[0].dispatchEvent(new Event('input', { bubbles: true })); s.call(i[1], 'admin123'); i[1].dispatchEvent(new Event('input', { bubbles: true }));
    [...document.querySelectorAll('button')].find(b => b.textContent.includes('เข้าสู่ระบบ')).click(); })()`);
  step(await waitFor(`location.pathname !== '/'`), 'login');

  await send('Page.navigate', { url: `${APP}/thai-frat-list` });
  await must(await waitFor(`document.querySelectorAll('.score-inputs select').length === 6`), 'the form shows six questions');
  await sleep(1500);   // let the list load, so the IDs that existed before this run are known
  const before = new Set(await ev(listIds) || []);
  await typeInto('input[placeholder="กรอกชื่อ-นามสกุล"]', NAME);
  await typeInto('input[type=tel]', '0800000000');
  await ev(`document.querySelector('.custom-dropdown').click()`);
  await waitFor(`document.querySelector('.dropdown-item')`, 5000);
  await clickText('.dropdown-item', 'ชลบุรี');
  step(await waitFor(`document.querySelector('.dropdown-selected')?.textContent.includes('ชลบุรี')`, 5000), 'province chosen from the dropdown');
  await ev(`(() => { const c = document.querySelector('#pdpa'); if (!c.checked) c.click(); })()`);
  const picked = [await answer(0, 25), await answer(1, 15), await answer(2, 15), await answer(3, 0), await answer(4, 10), await answer(5, 0)];
  step(picked.every(Boolean), 'answered all six questions', JSON.stringify(picked));
  await clickText('button', 'บันทึกข้อมูล');
  step(await waitFor(`location.pathname === '/thai-frat-list'`), 'saved, still on the list page');
  await must(await waitFor(row), 'the new person is in the list');
  const mine = await ev(`[...document.querySelectorAll('tbody tr')].filter(r => r.cells[0]?.textContent.trim() === ${JSON.stringify(NAME)})
    .map(r => (r.querySelector('a')?.getAttribute('href') || '').match(/thai-frat-detail\\/(\\d+)$/)?.[1])`);
  const MYID = mine && mine.length === 1 && mine[0] && !before.has(mine[0]) ? mine[0] : null;
  await must(!!MYID, 'exactly one row has this run’s name and its ID is new', JSON.stringify({ mine, existedBefore: mine?.map(i => before.has(i)) }));
  const onMine = `location.pathname === '/thai-frat-detail/${MYID}'`;
  const listScore = await ev(`(${row})?.querySelector('.score-display')?.textContent.trim()`);
  step(listScore === '65', 'list shows the right total (25+15+15+0+10+0)', listScore);

  await ev(`[...document.querySelectorAll('tbody tr a')].find(a => a.getAttribute('href') === '/thai-frat-detail/${MYID}').click()`);
  await must(await waitFor(onMine), 'opened "ดูรายละเอียด" of the record this run created', MYID);
  await waitFor(`document.body.innerText.includes(${JSON.stringify(NAME)})`);
  const detail = await ev(`document.body.innerText.match(/คะแนนรวม:\\s*(\\d+)/)?.[1] + ' | ' + (document.body.innerText.match(/ระดับความเสี่ยง:\\s*([^\\n]+)/)?.[1] || '')`);
  step(detail.startsWith('65 |') && detail.split('|')[1].trim().length > 0, 'detail shows the same total and a risk level', detail);

  await clickText('button', 'แก้ไข');
  await must(await waitFor(`location.pathname.startsWith('/thai-frat-form') && new URLSearchParams(location.search).get('edit') === '${MYID}' && document.querySelectorAll('.score-inputs select').length === 6`), 'edit opens the form for this record');
  await must(await waitFor(`document.querySelector('input[placeholder="กรอกชื่อ-นามสกุล"]')?.value === ${JSON.stringify(NAME)}`, 8000), 'the edit form holds this run’s record');
  await answer(0, 0);
  await clickText('button', 'อัปเดตข้อมูล');
  await must(await waitFor(onMine), 'updated -> back on this record’s detail page');
  const after = await (async () => { for (let i = 0; i < 20; i++) { const v = await ev(`document.body.innerText.match(/คะแนนรวม:\\s*(\\d+)/)?.[1]`); if (v === '40') return v; await sleep(500); }
    return ev(`document.body.innerText.match(/คะแนนรวม:\\s*(\\d+)/)?.[1]`); })();
  step(after === '40', 'answer 1 changed 25 -> 0: total 65 -> 40', after);

  await must(await ev(`!!(${onMine}) && document.body.innerText.includes(${JSON.stringify(NAME)})`), 'about to delete: the page is this run’s record', MYID);
  await ev(`window.confirm = () => true`);
  await clickText('button', 'ลบ');
  step(await waitFor(`location.pathname === '/thai-frat-list'`), 'deleted -> back on the list');
  await sleep(1500);
  const left = await ev(listIds) || [];
  step(!left.includes(MYID), 'the record this run created is gone from the list', MYID);
  step([...before].every(i => left.includes(i)), 'every record that existed before this run is still there', `${before.size} before`);
  await finish();
}
main().catch(e => { console.log('FAIL  crashed:', e.message); process.exit(1); });
