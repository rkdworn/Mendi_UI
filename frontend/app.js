'use strict';
const $ = id => document.getElementById(id);
const HEADERS = {'ngrok-skip-browser-warning': 'true'};
const LABELS = {normal:'정상', heel:'뒤꿈치 안감 손상', outsole:'밑창 손상', both:'안감 · 밑창 손상'};
let base = '', connected = false, lastFile = null, lastImg = null, lastResult = null, controller = null;

function openDrawer() { $('drawer').classList.add('on'); $('overlay').classList.add('on'); $('drawer').setAttribute('aria-hidden','false'); $('menuBtn').setAttribute('aria-expanded','true'); }
function closeDrawer() { $('drawer').classList.remove('on'); $('overlay').classList.remove('on'); $('drawer').setAttribute('aria-hidden','true'); $('menuBtn').setAttribute('aria-expanded','false'); }
function setStatus(message, kind) {
  $('status').textContent = message; $('status').className = kind;
  $('connectionNotice').textContent = message;
  $('dot').className = 'dot ' + (kind === 'wait' ? '' : kind);
  $('drop').classList.toggle('disabled', !connected);
}
function initialApi() {
  const query = new URLSearchParams(location.search).get('api');
  if (query) return query;
  if (!location.hostname.endsWith('.github.io')) return location.origin;
  try { return localStorage.getItem('mendaiDamageApi') || ''; } catch (_) { return ''; }
}
function validateUrl(value) {
  const url = new URL(value);
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && ['127.0.0.1','localhost'].includes(url.hostname))) throw new Error('HTTPS 서버 주소를 입력해 주세요.');
  if (url.username || url.password || url.search || url.hash) throw new Error('서버의 기본 주소만 입력해 주세요.');
  return (url.origin + url.pathname).replace(/\/+$/, '');
}
async function connect(value) {
  connected = false;
  if (!value.trim()) { setStatus('분석 서버 주소를 연결해 주세요.', 'wait'); return; }
  setStatus('분석 서버 연결 중…', 'wait');
  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), 15000);
  try {
    const candidate = validateUrl(value.trim());
    const response = await fetch(candidate + '/health', {headers:HEADERS, signal:abort.signal});
    if (!response.ok) throw new Error('서버 응답 오류: ' + response.status);
    const data = await response.json();
    if (data.status !== 'ok' || !data.thresholds || !data.artifact_sha256) throw new Error('손상 분석 서버 주소인지 확인해 주세요.');
    base = candidate; connected = true; $('api').value = base;
    setStatus('분석 서버 연결됨 · 사진을 올려 주세요.', 'ok');
    const li = text => { const item = document.createElement('li'); item.textContent = text; return item; };
    $('ftModel').replaceChildren(li('DINOv3 + RF-DETR + YOLO'),li('3개 모델 앙상블'));
    $('ftClasses').replaceChildren(...Object.values(LABELS).map(li));
    try { localStorage.setItem('mendaiDamageApi', base); } catch (_) {}
    closeDrawer();
  } catch (error) {
    setStatus(error.name === 'AbortError' ? '연결 시간이 초과됐습니다. 서버 실행 상태를 확인해 주세요.' : '연결 실패 · 서버 주소와 실행 상태를 확인해 주세요.', 'err');
  } finally { clearTimeout(timer); }
}
function loadImage(file) {
  return new Promise((resolve,reject) => {
    const url = URL.createObjectURL(file), image = new Image();
    image.onload = () => { URL.revokeObjectURL(url); resolve(image); };
    image.onerror = () => { URL.revokeObjectURL(url); reject(new Error('사진을 표시할 수 없습니다.')); };
    image.src = url;
  });
}
async function pick(file) {
  if (!connected) { openDrawer(); $('api').focus(); return; }
  if (!file || !file.type.startsWith('image/')) { setStatus('사진 파일을 선택해 주세요.', 'err'); return; }
  if (file.size > 20 * 1024 * 1024) { setStatus('20MB 이하의 사진을 선택해 주세요.', 'err'); return; }
  lastFile = file;
  if (controller) controller.abort();
  controller = new AbortController();
  const active = controller;
  const timer = setTimeout(() => active.abort(), 120000);
  $('resultCard').hidden = false; $('stage').classList.add('busy');
  $('summary').textContent = '사진을 분석하고 있어요'; $('meta').textContent = file.name;
  $('dets').replaceChildren(); $('scoreNote').hidden = true;
  const form = new FormData(); form.append('file', file);
  try {
    const [image,response] = await Promise.all([loadImage(file),fetch(base+'/predict',{method:'POST',headers:HEADERS,body:form,signal:active.signal})]);
    if (!response.ok) { let detail = '서버 오류: '+response.status; try {detail = (await response.json()).detail || detail;} catch (_) {} throw new Error(detail); }
    const result = await response.json();
    if (!(result.prediction in LABELS) || !result.scores || !result.thresholds) throw new Error('분석 결과 형식이 올바르지 않습니다.');
    for (const key of ['heel_lining','outsole']) if (!Number.isFinite(result.scores[key]) || result.scores[key]<0 || result.scores[key]>1 || !Number.isFinite(result.thresholds[key])) throw new Error('분석 점수 형식이 올바르지 않습니다.');
    if (active !== controller) return;
    lastImg = image; lastResult = result;
    render();
    $('resultCard').scrollIntoView({block:'nearest',behavior:'smooth'});
  } catch (error) {
    if (active !== controller) return;
    $('summary').textContent = '분석을 완료하지 못했어요';
    $('meta').textContent = error.name === 'AbortError' ? '응답 시간이 초과됐습니다. 서버 연결을 확인하고 다시 시도해 주세요.' : error.message;
    $('dets').replaceChildren();
  } finally {clearTimeout(timer); if(active===controller) $('stage').classList.remove('busy');}
}
function render() {
  const result = lastResult;
  $('summary').textContent = LABELS[result.prediction];
  $('summary').style.color = result.prediction === 'normal' ? 'var(--green)' : 'var(--orange-dark)';
  $('meta').textContent = lastFile.name + ' · 분석 ' + Number(result.inference_seconds || 0).toFixed(2) + '초';
  const list = $('dets'); list.replaceChildren();
  for(const [key,label,color] of [['heel_lining','뒤꿈치 안감','#fe7235'],['outsole','밑창','#3e976c']]) {
    const item = document.createElement('li');
    item.innerHTML = '<span class="sw"></span><span class="nm"></span><span class="bar"><i></i></span><span class="pct"></span>';
    item.querySelector('.sw').style.background = color;
    item.querySelector('.nm').textContent = label;
    item.querySelector('.bar > i').style.cssText = 'width:'+(result.scores[key]*100).toFixed(1)+'%;background:'+color;
    item.querySelector('.pct').textContent = result.scores[key] >= result.thresholds[key] ? '손상 감지' : '미감지';
    item.title = label+' 모델 점수 '+(result.scores[key]*100).toFixed(1)+'%';
    list.append(item);
  }
  $('scoreNote').hidden = false;
  draw();
}
function draw() {
  if(!lastImg) return;
  const canvas=$('canvas'), scale=Math.min(1,1600/Math.max(lastImg.naturalWidth,lastImg.naturalHeight));
  canvas.width=Math.round(lastImg.naturalWidth*scale); canvas.height=Math.round(lastImg.naturalHeight*scale);
  canvas.getContext('2d').drawImage(lastImg,0,0,canvas.width,canvas.height);
}
$('menuBtn').onclick=openDrawer; $('settingsBtn').onclick=() => {openDrawer();$('api').focus();};
$('openServer').onclick=() => {openDrawer();$('api').focus();};
$('closeBtn').onclick=closeDrawer; $('overlay').onclick=closeDrawer;
document.querySelectorAll('.drawer nav a').forEach(link => link.onclick=closeDrawer);
document.addEventListener('keydown',event => {if(event.key==='Escape')closeDrawer();});
$('connect').onclick=() => connect($('api').value);
$('api').addEventListener('keydown',event => {if(event.key==='Enter')connect($('api').value);});
$('drop').onclick=() => {if(connected)$('file').click();else openDrawer();};
$('drop').addEventListener('keydown',event => {if(event.key==='Enter'||event.key===' '){event.preventDefault();$('drop').click();}});
$('file').onchange=() => {pick($('file').files[0]);$('file').value='';};
for(const name of ['dragenter','dragover'])$('drop').addEventListener(name,event=>{event.preventDefault();$('drop').classList.add('over');});
for(const name of ['dragleave','drop'])$('drop').addEventListener(name,event=>{event.preventDefault();$('drop').classList.remove('over');});
$('drop').addEventListener('drop',event=>pick(event.dataTransfer.files[0]));
const toTool=()=>$('tool').scrollIntoView({behavior:'smooth'});
$('fab').onclick=toTool;$('ftUpload').onclick=toTool;$('totop').onclick=()=>window.scrollTo({top:0,behavior:'smooth'});
if('IntersectionObserver' in window)new IntersectionObserver(([entry])=>{$('fab').hidden=entry.isIntersecting;},{threshold:.3}).observe($('drop'));
window.addEventListener('resize',draw);
$('api').value=initialApi();connect($('api').value);
