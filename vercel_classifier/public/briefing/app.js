'use strict';
const $ = id => document.getElementById(id);
const escape = value => String(value ?? 'Unknown').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const table = (headers, rows) => '<table><thead><tr>'+headers.map(x=>'<th scope="col">'+escape(x)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+escape(x)+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
let data, queue=[], permits=[], page=0, map, roads;
const colors = ['#0069be','#da5f00','#be0041','#64199b'];
const color = days => colors[days<=30?0:days<=90?1:days<=365?2:3];
function activate(name) {
 for (const id of ['closures','research','population']) { $(id).hidden=id!==name; $({closures:'closure-tab',research:'research-tab',population:'population-tab'}[id]).setAttribute('aria-selected',String(id===name)); }
 if(name==='closures' && map) setTimeout(()=>map.invalidateSize(),100);
}
$('closure-tab').onclick=()=>activate('closures'); $('research-tab').onclick=()=>activate('research');
const panelFromHash=()=>['research','population'].includes(location.hash.slice(1))?location.hash.slice(1):'closures';
activate(panelFromHash());
window.addEventListener('hashchange',()=>activate(panelFromHash()));
$('population-tab').onclick=()=>activate('population');
if(new URLSearchParams(location.search).has('embed')) {
 document.querySelector('header').hidden=true;
 document.querySelector('nav').hidden=true;
 document.body.style.background='transparent';
 const reportHeight=()=>parent.postMessage({type:'briefing-height',height:document.documentElement.scrollHeight},location.origin);
 new ResizeObserver(reportHeight).observe(document.body);
}
function drawQueue() {
 const query=$('search').value.trim().toLowerCase();
 const rows=permits.filter(r=>[r.permit_id,r.primary_street,r.work_description].join(' ').toLowerCase().includes(query));
 const pages=Math.max(1,Math.ceil(rows.length/25)); page=Math.min(page,pages-1);
 $('queue').innerHTML=table(['Permit','Street','Work description','Recorded work type','End date','Days past due','Status'],rows.slice(page*25,(page+1)*25).map(r=>[r.permit_id,r.primary_street,r.work_description,r.work_type?.trim()||'Blank',r.end_date,r.days,r.active_norm==='null'?'Unknown':r.active_norm]));
 $('page').textContent=`Page ${page+1} of ${pages} · ${rows.length.toLocaleString()} permits`;
 $('prev').disabled=page===0; $('next').disabled=page===pages-1;
}
function drawGroups() {
 const key=$('group').value, counts=new Map();
 permits.forEach(r=>{const name=r[key]||'Unknown';counts.set(name,(counts.get(name)||0)+1);});
 $('groups').innerHTML=table(['Group','Distinct permits','Share of queue'],[...counts].sort((a,b)=>b[1]-a[1]).slice(0,20).map(([name,n])=>[name,n,(100*n/Math.max(permits.length,1)).toFixed(1)+'%']));
}
function update() {
 const date=new Intl.DateTimeFormat('en-CA',{timeZone:'America/New_York',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date());
 const today=Date.parse(date+'T00:00:00Z');
 let min=Number($('age').value); min=Number.isFinite(min)?Math.max(1,Math.min(3650,Math.round(min))):30; $('age').value=min;
 queue=data.records.map(r=>({...r,days:Math.floor((today-Date.parse(r.end_date+'T00:00:00Z'))/86400000)})).filter(r=>r.days>=min).sort((a,b)=>b.days-a.days);
 const seen=new Set(); permits=queue.filter(r=>{if(!r.permit_id?.trim()||seen.has(r.permit_id))return false;seen.add(r.permit_id);return true;});
 const ages=permits.map(r=>r.days).sort((a,b)=>a-b), n=ages.length;
 const median=n ? (n%2 ? ages[Math.floor(n/2)] : (ages[n/2-1]+ages[n/2])/2):null;
 const mapped=queue.filter(r=>r.path);
 $('asof').textContent=`As of ${date} · Pittsburgh time`;
 $('metrics').innerHTML=[['Past-due permits',n.toLocaleString()],['Mapped segments',mapped.length.toLocaleString()],['Median days past due',median===null?'—':Math.round(median).toLocaleString()],['Max days past due',n?ages[n-1].toLocaleString():'—']].map(([label,value])=>`<div><strong>${value}</strong><span>${label}</span></div>`).join('');
 $('mapnote').textContent=`Lines follow recorded GIS geometry. ${queue.length-mapped.length} qualifying segment rows could not be mapped. Endpoints are reported, not inferred.`;
 $('focus').innerHTML='<option value="">All qualifying closures</option>'+[...seen].sort().map(id=>`<option value="${escape(id)}">${escape(id)}</option>`).join('');
 if(map) {
  if(roads) map.removeLayer(roads);
  roads=L.featureGroup().addTo(map);
  mapped.forEach(r=>{
   const layer=L.polyline(r.path.map(([lon,lat])=>[lat,lon]),{color:color(r.days),weight:6});
   const popup=document.createElement('div');popup.style.whiteSpace='pre-line';popup.textContent=`${r.permit_id||'Unidentified permit'}\n${r.primary_street||'Unknown street'}\nFrom: ${r.from_street}\nTo: ${r.to_street}\n${r.days} days past recorded end`;
   layer.bindPopup(popup);layer.permitId=r.permit_id;layer.addTo(roads);
  });
  if(roads.getBounds().isValid()) map.fitBounds(roads.getBounds(),{padding:[30,30],maxZoom:17});
 }
 const old=permits.filter(r=>r.days>365).length;
 $('observations').textContent=`${n.toLocaleString()} distinct permits meet the filter across ${queue.length.toLocaleString()} segment rows. ${old.toLocaleString()} permits are more than a year past their recorded end date. Verify extensions and completion before treating these as ongoing obstructions.`;
 page=0;drawQueue();drawGroups();drawWorkTypes();
}
$('age').onchange=()=>update(); $('search').oninput=()=>{page=0;drawQueue();};
$('prev').onclick=()=>{page--;drawQueue();};$('next').onclick=()=>{page++;drawQueue();};$('group').onchange=drawGroups;
$('focus').onchange=()=>{if(!map||!roads)return;const layers=roads.getLayers().filter(l=>!$('focus').value||l.permitId===$('focus').value);const bounds=L.featureGroup(layers).getBounds();if(bounds.isValid())map.fitBounds(bounds,{padding:[35,35],maxZoom:17});};
$('download').onclick=()=>{
 const keys=['permit_id','primary_street','work_description','work_type','end_date','days','active_norm'];
 const cell=x=>'"'+String(x??'').replace(/^[=+@-]/,"'$&").replace(/"/g,'""')+'"';
 const text=[keys,...permits.map(r=>keys.map(k=>r[k]))].map(r=>r.map(cell).join(',')).join('\r\n');
 const url=URL.createObjectURL(new Blob([text],{type:'text/csv'}));const a=document.createElement('a');a.href=url;a.download='pittsburgh-permit-queue.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
fetch('snapshot.json').then(r=>{if(!r.ok)throw Error('snapshot');return r.json();}).then(payload=>{
 data=payload;
 if(typeof L!=='undefined') {map=L.map('map',{preferCanvas:true}).setView([40.4406,-79.9959],12);L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}).addTo(map);} else $('map').textContent='Map library could not load. The queue and research results remain available.';
 $('quality').innerHTML=table(['Data quality measure','Segment rows / permits'],Object.entries(data.quality).map(([k,v])=>[k.replaceAll('_',' '),v]));
 $('results').innerHTML=table(['Model','Prompt','Scored','Accuracy','Macro-F1','Failed responses'],data.results.map(r=>[r.model,r.prompt,r.n_scored,(100*r.accuracy).toFixed(2)+'%',r.macro_f1.toFixed(3),r.failed_predictions]));
 $('comparison').innerHTML='<h3>Accuracy by model and prompt</h3>'+[...new Set(data.results.map(r=>r.model))].map(model=>'<div class="bar-row"><strong>'+escape(model)+'</strong><div>'+data.results.filter(r=>r.model===model).map(r=>`<div class="bar ${r.prompt==='8 examples'?'old':''}" style="width:${100*r.accuracy}%">${escape(r.prompt)} · ${(100*r.accuracy).toFixed(1)}%</div>`).join('')+'</div></div>').join('');
 $('loading').hidden=true;update();
}).catch(()=>{$('loading').textContent='The saved snapshot could not load. Please refresh or try again later.';});

fetch('original-results.json').then(r=>{if(!r.ok)throw Error();return r.json();}).then(rows=>{
 $('original-results').innerHTML=table(['Model','Examples','Scored','Accuracy','Macro-F1','Failed responses'],rows.map(r=>[r.model,r.n_examples,r.n_scored,(100*r.accuracy).toFixed(2)+'%',r.macro_f1.toFixed(3),r.failed_predictions]));
}).catch(()=>{$('original-results').textContent='Original experiment summary could not load. Please refresh.';});
fetch('population.json').then(r=>{if(!r.ok)throw Error();return r.json();}).then(p=>{
 $('population-taxonomy').innerHTML='<ul>'+p.work_types.map(t=>'<li>'+escape(t)+'</li>').join('')+'</ul>';
 $('population-source').textContent=`The source contains ${p.source_rows.toLocaleString()} segment rows, with ${p.missing_rows.toLocaleString()} blank work-type rows (${(100*p.missing_rows/p.source_rows).toFixed(1)}%). There are ${p.unique_permits.toLocaleString()} identified permits, ${p.entirely_uncategorized.toLocaleString()} entirely uncategorized, and ${p.eligible_permits.toLocaleString()} eligible under the consistent-description rule. The 250 sampled permits span ${p.sample_segment_rows.toLocaleString()} segment rows.`;
 $('population-categories').innerHTML=table(['Proposed work type or abstention','Permits','Share of sample'],Object.entries(p.category_counts).map(([label,count])=>[label,count,(100*count/p.summary.total).toFixed(1)+'%']));
 $('population-permits').innerHTML=table(['Permit type','Sampled permits'],Object.entries(p.permit_types));
 $('population-confidence').innerHTML=table(['Minimum self-reported confidence','Assigned permits retained'],p.thresholds.map(r=>[r.threshold,r.sample_assigned_n]));
}).catch(()=>{$('population-source').textContent='Population evidence could not load. Please refresh.';});

function drawWorkTypes() {
 const blank=permits.filter(r=>!r.work_type?.trim());
 const eligible=blank.filter(r=>r.type_audit.classification_eligible);
 const saved=blank.filter(r=>r.saved_prediction);
 $('type-metrics').innerHTML=[['Recorded type present',permits.length-blank.length],['Blank on queue row',blank.length],['Eligible entirely blank permits',eligible.length],['Saved predictions for blanks',saved.length]].map(([label,n])=>`<div><strong>${n}</strong><span>${label}</span></div>`).join('');
 const counts=new Map();permits.forEach(r=>{const label=r.work_type?.trim()||'Blank';counts.set(label,(counts.get(label)||0)+1);});
 $('type-breakdown').innerHTML=table(['Recorded work type on queue row','Distinct permits'],[...counts].sort((a,b)=>b[1]-a[1]));
 $('type-summary').textContent=`${blank.length} of ${permits.length} filtered permits have a blank type on their oldest qualifying row. ${eligible.length} are entirely blank with usable, consistent descriptions across all source segments. ${saved.length} have matching saved full-taxonomy predictions. The remaining blank permits have no matching saved confidence score; mixed labels or inconsistent/missing descriptions require reconciliation before using the population workflow.`;
 $('blank-types').innerHTML=blank.length?table(['Permit','Permit type','Status','Source audit / readiness','Saved proposed type','Model confidence','Model reason'],blank.map(r=>{
  const p=r.saved_prediction;
  return [r.permit_id,r.permit_type,r.active_norm==='null'?'Unknown':r.active_norm,r.type_audit.classification_eligible?'Eligible for classification':r.type_audit.recorded_types.length?'Other segments already labeled: '+r.type_audit.recorded_types.join(', '):'Description needs review',p?(p.status==='ASSIGNED'?p.category:p.status):'Not assessed',p?.confidence!=null?p.confidence.toFixed(2):'Not assessed',p?.reason||'No matching saved prediction'];
 })): '<p>No blank work types in the current filter.</p>';
}
