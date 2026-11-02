import re
import os

with open('demo/static/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

with open('demo/static/script.js', 'r', encoding='utf-8') as f:
    js = f.read()

# Find all IDs in HTML
html_ids = set(re.findall(r'id=["\']([a-zA-Z0-9_\-]+)["\']', html))

# Find all getElementById in JS
js_ids = set(re.findall(r'getElementById\(["\']([a-zA-Z0-9_\-]+)["\']\)', js))

print('=== 2. Cross-Checking HTML Element IDs vs JavaScript Callers ===')
print(f'Total HTML IDs defined: {len(html_ids)}')
print(f'Total JS getElementById lookups: {len(js_ids)}')

missing_ids = js_ids - html_ids
if missing_ids:
    print(f'  [FAIL] Missing IDs in HTML that JS references: {missing_ids}')
else:
    print('  [PASS] All JavaScript DOM element lookups match valid HTML IDs!')

# Check for classes queried by querySelectorAll
classes_js = set(re.findall(r'querySelectorAll\(["\']\.([a-zA-Z0-9_\-]+)["\']\)', js))
html_classes = set(re.findall(r'class=["\']([^"\']+)["\']', html))
all_html_classes = set()
for c in html_classes:
    for single_c in c.split():
        all_html_classes.add(single_c)

missing_classes = classes_js - all_html_classes
if missing_classes:
    print(f'  [FAIL] Missing classes in HTML that JS queries: {missing_classes}')
else:
    print('  [PASS] All JavaScript class queries match valid HTML classes!')

