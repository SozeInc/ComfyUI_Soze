class Soze_IntToString:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "int": ("INT", {"default": 0, "min": -2147483648, "max": 2147483647, "step": 1}),
                "format": ("STRING", {"default": "" }),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "int_to_string"

    CATEGORY = "Soze Nodes"

    def int_to_string(self, int, format):
        if format:
            try:
                formatted_string = format.format(int)
            except Exception:
                formatted_string = str(int)
            return (formatted_string,)
        else:
            return (str(int),)
    


class Soze_StringToInt:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"default": "0"}),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "string_to_int"

    CATEGORY = "Soze Nodes"

    def string_to_int(self, string):
        try:
            int_value = int(string)
        except ValueError:
            int_value = 0
        return (int_value,)
    
class Soze_FloatToString:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float": ("FLOAT", {"default": 0.0, "min": -1e+20, "max": 1e+20, "step": 0.01}),
                "format": ("STRING", {"default": "" }),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "float_to_string"

    CATEGORY = "Soze Nodes"

    def float_to_string(self, float, format):
        if format:
            try:
                formatted_string = format.format(float)
            except Exception:
                formatted_string = str(float)
            return (formatted_string,)
        else:
            return (str(float),)
    
class Soze_StringToFloat:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"default": "0.0"}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "string_to_float"

    CATEGORY = "Soze Nodes"

    def string_to_float(self, string):
        try:
            float_value = float(string)
        except ValueError:
            float_value = 0.0
        return (float_value,)
    
class Soze_BoolToString:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bool": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "bool_to_string"

    CATEGORY = "Soze Nodes"

    def bool_to_string(self, bool):
        return (str(bool),)

class Soze_StringToBool:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"default": "False"}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "string_to_bool"

    CATEGORY = "Soze Nodes"

    def string_to_bool(self, string):
        bool_value = string.lower() in ("true", "1", "yes", "on")
        return (bool_value,)
    


class Soze_FloatToInt:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float": ("FLOAT", {"default": 0.0, "min": -1e+20, "max": 1e+20, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "float_to_int"

    CATEGORY = "Soze Nodes"

    def float_to_int(self, float):
        return (int(float),)
    
class Soze_IntToFloat:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "int": ("INT", {"default": 0, "min": -2147483648, "max": 2147483647, "step": 1}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "int_to_float"

    CATEGORY = "Soze Nodes"

    def int_to_float(self, int):
        return (float(int),)
    
    
class Soze_BooleanInverter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bool": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "invert_boolean"

    CATEGORY = "Soze Nodes"

    def invert_boolean(self, bool):
        return (not bool,)
