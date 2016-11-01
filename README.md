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
# from PIL import Image

fig = plt.figure()
axis = [fig.add_subplot(121 + _) for _ in range(2)]
bm = xpm.XPM('/tmp/testdata.xpm') # as ndarray
axis[0].imshow(bm)
misc.imsave('/tmp/testdata.gif', np.uint8(bm))
misc.imsave('/tmp/testdata.gif', misc.bytescale(bm, cmin=0, cmax=255))
im = misc.toimage(bm, cmin=0, cmax=255) # same as PIL.Image
axis[1].imshow(im)
im.save('/tmp/testdata.png')
plt.show()
```


Links
-----

github https://github.com/sanjonemu/pyxpm

pyxpm https://pypi.python.org/pypi/pyxpm


License
-------

MIT License

