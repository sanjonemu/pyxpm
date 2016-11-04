pyxpm
=====

XPM image file loader for Python (to numpy ndarray or PIL) native C .pyd


How to use
----------

```python
from pyxpm import xpm
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from scipy import misc
from PIL import Image

fig = plt.figure()
axis = [fig.add_subplot(211 + _) for _ in range(2)]
s = open('/tmp/testdata.xpm', 'rb').read()
nda = xpm.XPM(s) # as ndarray (dtype=np.uint8) BGR(A)
r, c, m = nda.shape
img = Image.frombuffer('RGBA', (c, r), nda, 'raw', 'BGRA', 0, 1)
img.show() # PIL.Image
bm = np.array(img) # RGB(A)
axis[0].imshow(bm)
misc.imsave('/tmp/testdata_0.gif', np.uint8(bm))
misc.imsave('/tmp/testdata_1.gif', misc.bytescale(bm, cmin=0, cmax=255))
im = misc.toimage(bm, cmin=0, cmax=255) # same as PIL.Image
im.save('/tmp/testdata.png')
axis[1].imshow(im)
plt.show()
```


Links
-----

github https://github.com/sanjonemu/pyxpm

pyxpm https://pypi.python.org/pypi/pyxpm


License
-------

MIT License

