def find_duplicate(items):
    seen = []
    for item in items:
        if item in seen:
            return item
        seen.append(item)
    return None
