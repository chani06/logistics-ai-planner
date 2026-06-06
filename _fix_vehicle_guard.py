content = open('c:/Users/chani/app/app.py', encoding='utf-8').read()
lines = content.split('\n')

# Lines 4212, 4213, 4214 (0-indexed: 4211, 4212, 4213)
l_continue = lines[4211]   # '                                continue'
l_force    = lines[4212]   # safe_print FORCE
l_append   = lines[4213]   # trip_codes.append

old_block = l_continue + '\n' + l_force + '\n' + l_append

vehicle_guard = (
    "                        # \U0001f6ab vehicle constraint guard\n"
    "                        _sg_test_codes_chk = (trip_codes if trip_codes else [start_code]) + [actual_code]\n"
    "                        _sg_allowed_chk = get_allowed_from_codes(_sg_test_codes_chk, ['4W', 'JB', '6W'])\n"
    "                        if not _sg_allowed_chk:\n"
    "                            safe_print(f\"      \U0001f6d1 START-GROUP VEH: \u0e15\u0e31\u0e14 {actual_code} vehicle constraint \u0e02\u0e31\u0e14\u0e41\u0e22\u0e49\u0e07 (\u0e2b\u0e48\u0e32\u0e07 {_gc_dist_s:.1f}km)\")\n"
    "                            continue\n"
)

new_block = l_continue + '\n' + vehicle_guard + l_force + '\n' + l_append

if old_block in content:
    content = content.replace(old_block, new_block, 1)
    open('c:/Users/chani/app/app.py', 'w', encoding='utf-8').write(content)
    print('OK: START_GROUP vehicle guard added')
else:
    print('NOT FOUND')
    print(repr(old_block[:120]))
