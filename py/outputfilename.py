import datetime
import re


def replace_date_placeholders(text):
    now = datetime.datetime.now()
    
    def replace_match(match):
        date_format = match.group(1)
        # Replace custom format with strftime format
        date_format = date_format.replace('yyyy', '%Y').replace('MM', '%m').replace('dd', '%d')
        date_format = date_format.replace('hh', '%H').replace('mm', '%M').replace('ss', '%S')
        return now.strftime(date_format)
    
    return re.sub(r'%date:([^%]+)%', replace_match, text)

class OutputFilename:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'path_delimiter': ('STRING', {'default': '/'}),
                             'file_delimiter': ('STRING', {'default': '-'})
                             },
                'optional': {'path_input_1': ('STRING', {'default': ''}),
                             'path_input_2': ('STRING', {'default': ''}),
                             'path_input_3': ('STRING', {'default': ''}),
                             'path_input_4': ('STRING', {'default': ''}),
                             'path_input_5': ('STRING', {'default': ''}),
                             'file_input_1': ('STRING', {'default': ''}),
                             'file_input_2': ('STRING', {'default': ''}),
                             'file_input_3': ('STRING', {'default': ''}),
                             'file_input_4': ('STRING', {'default': ''}),
                             'file_input_5': ('STRING', {'default': ''})
                             }}

    RETURN_NAMES = ('FileNamePath','Path','FileName')
    RETURN_TYPES = ('STRING','STRING','STRING')
    FUNCTION = 'OutputFilename'
    CATEGORY = 'Soze/strings'

    def OutputFilename(self, path_delimiter, path_input_1, path_input_2, path_input_3, path_input_4, path_input_5, file_delimiter, file_input_1, file_input_2, file_input_3, file_input_4, file_input_5):
        paths = []
        if path_input_1:
            paths.append(replace_date_placeholders(path_input_1))
        if path_input_2:
            paths.append(replace_date_placeholders(path_input_2))
        if path_input_3:
            paths.append(replace_date_placeholders(path_input_3))
        if path_input_4:
            paths.append(replace_date_placeholders(path_input_4))
        if path_input_5:
            paths.append(replace_date_placeholders(path_input_5))
        path = path_delimiter.join(paths)
        files = []
        if file_input_1:
            files.append(replace_date_placeholders(file_input_1))
        if file_input_2:
            files.append(replace_date_placeholders(file_input_2))
        if file_input_3:
            files.append(replace_date_placeholders(file_input_3))
        if file_input_4:
            files.append(replace_date_placeholders(file_input_4))
        if file_input_5:
            files.append(replace_date_placeholders(file_input_5))
        files = file_delimiter.join(files)
        filepath = path_delimiter.join([path,files])
        return (filepath, path,files,)

    
