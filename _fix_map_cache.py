with open(r'c:\Users\chani\app\app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f'Starting line count: {len(lines)}')

INDENT = '                                            '  # 44 spaces

# --- Insert cache check lines before line 4899 (0-indexed: 4898) ---
cache_check_lines = [
    INDENT + '# Map cache - ตรวจสอบว่ามีแผนที่ cached หรือไม่\n',
    INDENT + "map_cache_key = f'map|{selected_trip}|{selected_truck}|{show_route}|{len(valid_coords)}'\n",
    INDENT + "_map_is_cached = (st.session_state.get('_map_cache_key') == map_cache_key and '_map_html' in st.session_state)\n",
    INDENT + 'if not _map_is_cached:\n',
]

# --- Indent lines 4899-5102 (0-indexed) by 4 spaces (inside 'if not _map_is_cached:') ---
for i in range(4898, 5103):
    if lines[i].strip():  # non-empty lines
        lines[i] = '    ' + lines[i]
    # empty lines stay empty

# Verify folium_static is at the expected position after re-indent (still index 5102)
print(f'folium_static line (index 5102): {repr(lines[5102].rstrip()[:80])}')

# --- Insert cache-save lines AFTER folium_static (index 5102) ---
cache_save_lines = [
    '\n',
    INDENT + '    # บันทึก Map Cache\n',
    INDENT + "    st.session_state['_map_cache_key'] = map_cache_key\n",
    INDENT + "    st.session_state['_map_html'] = m._repr_html_()\n",
    '\n',
    INDENT + '# แสดงแผนที่จาก cache เมื่อ cached\n',
    INDENT + "if _map_is_cached and '_map_html' in st.session_state:\n",
    INDENT + '    import streamlit.components.v1 as _cmp\n',
    INDENT + "    _cmp.html(st.session_state['_map_html'], height=720, scrolling=False)\n",
]
lines = lines[:5103] + cache_save_lines + lines[5103:]

# --- Now insert cache check lines before the spinner (index 4898) ---
lines = lines[:4898] + cache_check_lines + lines[4898:]

print(f'Inserted {len(cache_check_lines)} cache-check lines at 4898')
print(f'Inserted {len(cache_save_lines)} cache-save lines after folium_static')
print(f'Final line count: {len(lines)}')

with open(r'c:\Users\chani\app\app.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('Written successfully')
