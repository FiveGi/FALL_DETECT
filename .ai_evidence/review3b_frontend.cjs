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
  vm.createContext(ctx); vm.runInContext(core+handlers+`;this.api={resolveUrl,handleError,clearRefresh,unmount:()=>{unmounted=true;requestSeq++;clearRefresh()},getUrl:()=>processedUrl.value};`,ctx);
  return {ctx,timers,pending,events,requests:()=>requests,api:ctx.api};
}
(async()=>{
 const x=setup(); await x.api.resolveUrl();
 for(let i=0;i<3;i++){x.api.handleError('mjpeg');const [id,t]=x.timers.entries().next().value;assert.equal(t.ms,1000*2**i);x.timers.delete(id);await t.fn();}
 x.api.handleError('mjpeg');assert.equal(x.timers.size,0);assert.equal(x.requests(),4);
 assert(x.events.find(e=>e.name==='error').event.url.endsWith('t=REDACTED'));
 console.log('PASS terminal retry stops after 3 retries; emitted error URL redacted');
 const y=setup(true),a=y.api.resolveUrl(),b=y.api.resolveUrl();y.pending[1]({url:'new',maxViewSeconds:600});await b;y.pending[0]({url:'stale',maxViewSeconds:600});await a;assert.equal(y.api.getUrl(),'new');assert.equal(y.timers.size,1);
 console.log('PASS same-camera out-of-order response ignored; one timer');
 const z=setup(true),f=z.api.resolveUrl();z.api.unmount();z.pending[0]({url:'late',maxViewSeconds:600});await f;assert.equal(z.timers.size,0);assert.equal(z.api.getUrl(),'');
 console.log('PASS unmount invalidation (callback-equivalent fixture); no late URL/timer');
 console.log('Synthetic extracted functions/fake HTTP and timers; 0 clips; no browser');
})().catch(e=>{console.error(e);process.exitCode=1});
