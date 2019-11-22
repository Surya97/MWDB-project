__all__ = ['storage']

def storage(storage_config, index):
    if 'dict' in storage_config:
        return InMemoryStorage(storage_config['dict'])

class BaseStorage(object):
    def __init__(self, config):
       raise NotImplementedError
    def keys(self):
        raise NotImplementedError
    def set_val(self, key, val):
       raise NotImplementedError
    def get_val(self, key):
        raise NotImplementedError
    def append_val(self, key, val, image_id=''):
       raise NotImplementedError
    def get_list(self, key):
        raise NotImplementedError

class InMemoryStorage(BaseStorage):
    def __init__(self, config):
        self.name = 'dict'
        self.storage = dict()

    def keys(self):
        return self.storage.keys()

    def set_val(self, key, val):
        self.storage[key] = val

    def get_val(self, key):
        return self.storage[key]

    def append_val(self, key, val, image_id = ''):
        self.storage.setdefault(key, []).append((val, image_id))

    def get_list(self, key):
        return self.storage.get(key, [])