import string

def replace_context(text, replObj):
    for key, value in replObj.items():
        if key != None and value != None:
            searchPattern = '<'+key+'>'
            if text.find(searchPattern) > 0:
                text = text.replace(searchPattern, '*'+str(value)+'*')
    return text

def remove_newline(text):
    return text.replace('\n', ' ')

def proper_noun(text):
    return string.capwords(text)
