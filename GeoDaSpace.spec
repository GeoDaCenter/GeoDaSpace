# -*- mode: python -*-
a = Analysis(['geodaspace/GeoDaSpace.py'],
             pathex=['c:\\Users\\stephens\\Documents\\spreg\\trunk'],
             hiddenimports=['scipy.special._ufuncs_cxx', 'scipy.sparse.csgraph._validation', 'scipy.io.matlab.streams'],
             hookspath=None,
             runtime_hooks=None)
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='GeoDaSpace.exe',
          debug=False,
          strip=None,
          upx=True,
          console=False , icon='geodaspace\\icons\\geodaspace.ico')
