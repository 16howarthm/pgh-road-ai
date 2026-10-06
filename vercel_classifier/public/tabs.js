'use strict';
const frame=document.getElementById('briefing-frame');
function selectTab(name) {
 for(const tab of ['classifier','closures','research','population']) document.getElementById('tab-'+tab).setAttribute('aria-selected',String(tab===name));
 document.getElementById('classifier-panel').hidden=name!=='classifier';
 const panel=document.getElementById('briefing-panel');panel.hidden=name==='classifier';panel.setAttribute('aria-labelledby','tab-'+name);
 if(name!=='classifier') frame.src='/briefing/index.html?embed=1#'+name;
 history.replaceState(null,'','#'+name);
}
for(const name of ['classifier','closures','research','population']) document.getElementById('tab-'+name).onclick=()=>selectTab(name);
window.addEventListener('message',event=>{if(event.origin===location.origin&&event.source===frame.contentWindow&&event.data?.type==='briefing-height'&&Number.isFinite(event.data.height))frame.style.height=Math.min(50000,Math.max(800,event.data.height+20))+'px';});
if(['closures','research','population'].includes(location.hash.slice(1)))selectTab(location.hash.slice(1));
