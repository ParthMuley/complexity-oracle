def running_total(items):
    total = 0
    result = []
    for item in items:
        total += item
        result.append(total)
    return result