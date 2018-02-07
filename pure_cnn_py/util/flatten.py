def flatten(alist):
    new_list = []
    for item in alist:
        if isinstance(item, (list, tuple)):
            new_list.extend(flatten(item))
        else:
            new_list.append(item)
    return new_list
