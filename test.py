import pylsl
print("Looking for all available streams...")
streams = pylsl.resolve_streams()
for s in streams:
    print(f"Found: {s.name()} (Type: {s.type()})")