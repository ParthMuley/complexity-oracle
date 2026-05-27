def count_common_elements(list_a, list_b):
    count = 0
    for item in list_a:
        if item in list_b:
            count += 1
    return count