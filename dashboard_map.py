"""Leaflet map using independent GeoJSON LineStrings from the source geometry."""
import json


def closure_features(map_df):
    features = []
    for row in map_df.to_dict('records'):
        features.append({
            'type': 'Feature',
            'geometry': {'type': 'LineString', 'coordinates': row['path']},
            'properties': {
                **{key: str(row.get(key, 'Unknown')) for key in
                   ['permit_id', 'primary_street', 'from_street', 'to_street', 'active_norm']},
                'days_past_due': int(row['days_past_due']),
                'color': '#%02x%02x%02x' % tuple(row['color'][:3]),
            },
        })
    return {'type': 'FeatureCollection', 'features': features}


def map_html(map_df, focus_permit=None):
    # Escape script delimiters in CSV-derived values. Tooltip content uses textContent.
    payload = json.dumps(closure_features(map_df), allow_nan=False).replace('<', '\\u003c')
    focus = json.dumps(focus_permit).replace('<', '\\u003c')
    return '''<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>html,body,#map{height:100%;margin:0}#status{position:absolute;z-index:1000;top:10px;left:55px;background:white;padding:8px;font:14px sans-serif}</style>
</head><body><div id="map"></div><div id="status">Loading closure map…</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const data = ''' + payload + ''';
const focusPermit = ''' + focus + ''';
const status = document.getElementById('status');
if (typeof L === 'undefined') {
  status.textContent = 'Map library could not load. Check your connection to unpkg.com.';
} else {
  const map = L.map('map', {preferCanvas:false}).setView([40.4406,-79.9959],12);
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom:19,
    attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  }).addTo(map);
  L.geoJSON(data, {style:{color:'#ffffff',weight:11,opacity:1},interactive:false}).addTo(map);
  const roads = L.geoJSON(data, {
    style: feature => ({color:feature.properties.color,weight:7,opacity:1,lineCap:'round',lineJoin:'round'}),
    onEachFeature: (feature, layer) => {
      const p = feature.properties;
      const text = document.createElement('div');
      text.style.whiteSpace = 'pre-line';
      text.textContent = p.permit_id + '\\n' + p.primary_street + '\\nFrom: ' + p.from_street + '\\nTo: ' + p.to_street + '\\n' + p.days_past_due + ' days past recorded end\\nActive: ' + p.active_norm;
      layer.bindTooltip(text, {sticky:true});
      layer.bindPopup(text.cloneNode(true));
    }
  }).addTo(map);
  const focused = data.features.filter(f => f.properties.permit_id === focusPermit);
  const bounds = focused.length ? L.geoJSON(focused).getBounds() : roads.getBounds();
  if (bounds.isValid()) map.fitBounds(bounds, {padding:[45,45],maxZoom:17});
  status.remove();
  setTimeout(() => map.invalidateSize(), 100);
}
</script></body></html>'''
