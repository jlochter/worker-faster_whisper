INPUT_VALIDATIONS = {
    'audio': {
        'type': str,
        'required': False,
        'default': None
    },
    'audio_base64': {
        'type': str,
        'required': False,
        'default': None
    },
    'model': {
        'type': str,
        'required': False,
        'default': 'large-v2'
    },
    'language': {
        'type': str,
        'required': False,
        'default': 'ja'
    },
    'chunk_length': {
        'type': int,
        'required': False,
        'default': 30
    },
}
