// Executes unchanged function bodies from MediaViewer.vue with fake timers and deferred HTTP.
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const source = fs.readFileSync('frontend/src/components/common/MediaViewer.vue', 'utf8');
const core = source.slice(source.indexOf("const processedUrl = ref('')"), source.indexOf('watch(() => [props.url'));
const handlers = source.slice(source.indexOf('function handleLoadStart('), source.indexOf('function handleMetadataLoaded('));
function setup(deferred=false) {
  const timers = new Map(), pending = [], events = []; let seq=0, requests=0;
  const ctx = {props:{url:'http://camera/video',cameraId:14,overlay:false,useStreamApi:true},
    ref:value=>({value}), isLoading:{},hasError:{},errorMessage:{},showLocalPathWarning:{},
    emit:(name,event)=>events.push({name,event}), isLocalPath:()=>false, convertToServerUrl:x=>x,
    setTimeout:(fn,ms)=>{timers.set(++seq,{fn,ms});return seq},clearTimeout:id=>timers.delete(id),
    streamService:{getCameraStreamUrl:()=>{requests++;return deferred ? new Promise(r=>pending.push(r)) : Promise.resolve({url:'http://api/stream?t='+requests,maxViewSeconds:600})}}};
  vm.createContext(ctx); vm.runInContext(core+handlers+`;this.api={resolveUrl,handleError,clearRefresh,getUrl:()=>processedUrl.value};`,ctx);
  return {ctx,timers,pending,events,requests:()=>requests,api:ctx.api};
}
(async()=>{
  const x=setup();await x.api.resolveUrl();
  for(let i=0;i<7;i++) {
    x.api.handleError('mjpeg');
    const [id,timer]=x.timers.entries().next().value;
    console.log('Error',i+1,'next retry ms=',timer.ms,'requests=',x.requests());
    x.timers.delete(id);await timer.fn();
  }
  assert(x.requests()===8);
  console.log('BUG exhausted retries: requests keep recurring at 570000 ms without any successful load.');
  const monitor = fs.readFileSync('frontend/src/views/MonitorView.vue','utf8');
  const callback = monitor.slice(monitor.indexOf('function handleMediaError('),monitor.indexOf('// Translate detection result'));
  let logged;
  vm.runInNewContext(callback+';handleMediaError(camera,event)', {
    camera:{name:'probe'},event:x.events.find(e=>e.name==='error').event,
    console:{error:(...args)=>logged=args.join(' ')}
  });
  assert(logged.includes('?t='));
  console.log('BUG browser log: actual MonitorView error callback logs token-bearing event.url.');
  const y=setup(true), first=y.api.resolveUrl(), second=y.api.resolveUrl();
  y.pending[1]({url:'new',maxViewSeconds:600});await second;
  y.pending[0]({url:'stale',maxViewSeconds:600});await first;
  assert(y.api.getUrl()==='stale' && y.timers.size===2);
  console.log('BUG same-camera race: older response overwrites newer URL; two timers remain.');
  const z=setup(true), flight=z.api.resolveUrl();
  z.api.clearRefresh(); // exact onUnmounted callback
  z.pending[0]({url:'after-unmount',maxViewSeconds:600});await flight;
  assert(z.timers.size===1);
  console.log('BUG unmount: in-flight response schedules a new refresh timer after cleanup.');
  console.log('Dataset: fake HTTP/timers, extracted actual functions; 0 clips; not a browser integration test.');
})().catch(e=>{console.error(e);process.exitCode=1});
