SEPARATOR_CHAR = '~'
SEPARATOR = ' ' + SEPARATOR_CHAR + ' '

def dataset_mapping(x):
    return {
        "x": x["text"],
        "y": x["fake"],
    }


def dataset_mapping_pairs(x):
    return {
        "x": x["text1"] + SEPARATOR + x['text2'],
        "y": x["fake"],
    }