import unittest
import pandas as pd
from dashboard_map import closure_features, map_html
from dashboard_data import DATA_PATH, clean, parse_path


class MapTests(unittest.TestCase):
    def test_example_permit_uses_active_source_geometry(self):
        data = clean(pd.read_csv(DATA_PATH), today='2026-09-24')
        rows = data[data.past_due_schedule & data.permit_id.eq('DOMI-OP-2025-09779')].copy()
        rows['path'] = rows.geometry.apply(parse_path)
        rows['color'] = [[190, 0, 65]] * len(rows)
        features = closure_features(rows)['features']
        self.assertEqual(len(features), 1)
        feature = features[0]
        self.assertEqual(feature['properties']['active_norm'], 't')
        self.assertEqual(feature['properties']['from_street'], 'WYONA WAY')
        self.assertEqual(feature['properties']['to_street'], 'STONELEA ST')
        self.assertEqual(feature['geometry']['coordinates'], [
            [-80.04655881,40.48371911], [-80.04643488,40.48397403],
            [-80.04630787,40.48423089], [-80.04629232,40.48426269]])

    def test_independent_features_and_safe_tooltip_data(self):
        paths = [[[-80,40.44],[-80.001,40.441]], [[-79.95,40.46],[-79.951,40.461]]]
        rows = pd.DataFrame([dict(path=p, color=[190,0,65], permit_id='</script><script>bad()</script>',
                                 days_past_due=40) for p in paths])
        features = closure_features(rows)['features']
        self.assertEqual([f['geometry']['coordinates'] for f in features], paths)
        html = map_html(rows)
        self.assertNotIn('</script><script>bad()', html)
        self.assertIn('text.textContent', html)
