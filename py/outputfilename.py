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

class Soze_OutputFilename:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'Path_Delimiter': ('STRING', {'default': '/'}),
                             'File_Delimiter': ('STRING', {'default': '-'}),
                             },
                'optional': {'Path_Input_1': ('STRING', {'default': '.'}),
                             'Path_Input_2': ('STRING', {'default': 'ProjectName', 'isInput': True }),
                             'Path_Input_3': ('STRING', {'default': '%date:yyyy-MM-dd%'}),
                             'Path_Input_4': ('STRING', {'default': 'GenerationType'}),
                             'Path_Input_5': ('STRING', {'default': ''}),
                             'File_Input_1': ('STRING', {'default': '', 'isInput': True }),
                             'File_Input_2': ('STRING', {'default': ''}),
                             'File_Input_3': ('STRING', {'default': ''}),
                             'File_Input_4': ('STRING', {'default': ''}),
                             'File_Input_5': ('STRING', {'default': ''})
                             }}

    RETURN_NAMES = ('Filename_&_Path','Path','Filename')
    RETURN_TYPES = ('STRING','STRING','STRING')
    FUNCTION = 'OutputFilename'
    CATEGORY = 'strings'

    def OutputFilename(self, Path_Delimiter, Path_Input_1, Path_Input_2, Path_Input_3, Path_Input_4, Path_Input_5, File_Delimiter, File_Input_1, File_Input_2, File_Input_3, File_Input_4, File_Input_5):
        paths = []
        if Path_Input_1:
            paths.append(replace_date_placeholders(Path_Input_1))
        if Path_Input_2:
            paths.append(replace_date_placeholders(Path_Input_2))
        if Path_Input_3:
            paths.append(replace_date_placeholders(Path_Input_3))
        if Path_Input_4:
            paths.append(replace_date_placeholders(Path_Input_4))
        if Path_Input_5:
            paths.append(replace_date_placeholders(Path_Input_5))
        path = Path_Delimiter.join(paths)
        files = []
        if File_Input_1:
            files.append(replace_date_placeholders(File_Input_1))
        if File_Input_2:
            files.append(replace_date_placeholders(File_Input_2))
        if File_Input_3:
            files.append(replace_date_placeholders(File_Input_3))
        if File_Input_4:
            files.append(replace_date_placeholders(File_Input_4))
        if File_Input_5:
            files.append(replace_date_placeholders(File_Input_5))
        files = File_Delimiter.join(files)
        filepath = Path_Delimiter.join([path,files])
        return (filepath, path,files,)

    
