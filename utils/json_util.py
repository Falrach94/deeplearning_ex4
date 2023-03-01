

def list_to_json(list):
    return '[' + ', '.join(list) + ']'


def serlist_to_json(list):
    return '[' + ', '.join([obj.to_json() for obj in list]) + ']'


def to_str(item):
    if type(item) == str:
        return f'"{item}"'
    if hasattr(item, "to_json"):
        return item.to_json()
    return str(item)

def dic_to_json(names, data):
    res = ''
    res += f'"{names[0]}":{to_str(data[0])}'
    for i in range(len(names)-1):
        res += f', "{names[i+1]}":{to_str(data[i+1])}'
    return res

