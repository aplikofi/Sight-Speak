import sys

class DummyCV2:
    def __getattr__(self, name):
        if name == 'VideoCapture':
            class DummyVideo:
                def read(self): return False, None
                def isOpened(self): return False
                def release(self): pass
            return DummyVideo
        if name.startswith('COLOR_') or name.startswith('LINE_') or name.startswith('FONT_'):
            return 0
        return lambda *args, **kwargs: None

sys.modules[__name__] = DummyCV2()
