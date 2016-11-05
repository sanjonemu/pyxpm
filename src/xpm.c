/*
  xpm.c

  >mingw32-make -f makefile.tdmgcc64
  >test_xpm.py

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# img = Image.open(...) # or create any PIL.Image
# nda = np.asarray(img) # instead of np.array(img)
# nda.flags.writeable = True # to change bytes like img.putpixel((x, y), c)

# RGBA
nda = np.array([
[[0xFF, 0xFF, 0xFF, 0xFF], [0x00, 0x00, 0xFF, 0xFF], [0xFF, 0x00, 0x00, 0xFF]],
[[0xFF, 0xFF, 0xFF, 0xFF], [0x00, 0x00, 0xFF, 0xFF], [0xFF, 0x00, 0x00, 0xFF]]
], dtype=np.uint8)
Image.fromarray(nda, 'RGBA').show()

# change color order BGRA -> RGBA OK
Image.frombuffer('RGBA', (3, 2), nda, 'raw', ('BGRA', 0, 1)).show()

# change r,c -> c,r OK
Image.frombuffer('RGBA', (2, 3), nda, 'raw', ('BGRA', 0, 1)).show()

Image.fromstring('RGBA', (3, 2), '''\
\xFF\xFF\xFF\xFF\x00\x00\xFF\xFF\xFF\x00\x00\xFF\
\xFF\xFF\xFF\xFF\x00\x00\xFF\xFF\xFF\x00\x00\xFF\
''').show()

# RGBA
a = [[[ 255, 255, 255, 255],[ 255,   0,   0, 255]],
     [[ 255, 255,   0, 255],[   0, 255, 255, 255]],
     [[   0, 255,   0,0xCC],[   0,   0,   0,0xCC]]]
# same order as above
b = '''\
\xFF\xFF\xFF\xFF\xFF\x00\x00\xFF\
\xFF\xFF\x00\xFF\x00\xFF\xFF\xFF\
\x00\xFF\x00\xCC\x00\x00\x00\xCC\
'''

fig = plt.figure()
axis = [fig.add_subplot(221 + _) for _ in range(4)]

# BGRA b<->t # same src
axis[0].imshow(Image.fromstring('RGBA', (3, 2), b, 'raw', 'BGRA')) # c=3, r=2

# RGBA b<->t # same src
axis[1].imshow(Image.fromstring('RGBA', (3, 2), b)) # c=3, r=2

w = np.array(Image.fromstring('RGBA', (3, 2), b)) # c=3, r=2 # same as axis[1]
# array([[[255, 255, 255, 255], [255,   0,   0, 255], [255, 255,   0, 255]],
#        [[  0, 255, 255, 255], [  0, 255,   0, 204], [  0,   0,   0, 204]]
#   ], dtype=uint8)
# skip

x = np.array(Image.fromstring('RGBA', (2, 3), b)) # c=2, r=3 # to axis[2]
# array([[[255, 255, 255, 255], [255,   0,   0, 255]],
#        [[255, 255,   0, 255], [  0, 255, 255, 255]],
#        [[  0, 255,   0, 204], [  0,   0,   0, 204]]
#   ], dtype=uint8)
axis[2].imshow(x) # r=3, c=2

y = np.array(Image.fromstring('RGBA', (2, 3), b).getdata()) # c=6, r=1 no dtype
# array([[255, 255, 255, 255], [255,   0,   0, 255], [255, 255,   0, 255],
#        [  0, 255, 255, 255], [  0, 255,   0, 204], [  0,   0,   0, 204]])
# skip

z = np.array(a, dtype=np.uint8) # r=3, c=2 # to axis[3]
# array([[[255, 255, 255, 255], [255,   0,   0, 255]],
#        [[255, 255,   0, 255], [  0, 255, 255, 255]],
#        [[  0, 255,   0, 204], [  0,   0,   0, 204]]
#   ], dtype=uint8)
axis[3].imshow(z) # r=3, c=2

plt.show()
```
*/

#define __XPM_MAKE_DLL_
#include <xpm.h>

#include <numpy/arrayobject.h>

#define DEBUGLOG 0
#define TESTLOG "../dll/_test_dll_.log"

static XPMCOLORMAP xpmcolors[] = { // (uint *) (A)RGB L.E. -> (BYTE *) B,G,R,A
  {0x08000000, "black"},
  {0x1E00007F, "dblue"},
  {0x2D007F00, "dgreen"},
  {0x3C007F7F, "dcyan"},
  {0x4B7F0000, "dred"},
  {0x5A7F007F, "dmagenta"},
  {0x697F7F00, "dyellow"},
  {0x787F7F7F, "gray"},
  {0x873F3F3F, "dgray"},
  {0x960000FF, "blue"},
  {0xA500FF00, "green"},
  {0xB400FFFF, "cyan"},
  {0xC3FF0000, "red"},
  {0xD2FF00FF, "magenta"},
  {0xE1FFFF00, "yellow"},
  {0xF7FFFFFF, "white"},
  {0x293F7F00, "darkolivegreen"}, // test
  {0xB100FFFF, "lightblue"}, // cyan
  {0x807F7F7F, "gray50"}, // test
  {0x70B2B2B2, "gray70"}, // test
  {0x7FD8D8D8, "gray85"}, // test
  {0x7EBFBFBF, "lightgray"}}; // dummy

static int rprintf(char *fmt, ...)
{
  char buf[BUFSIZE];
  va_list va;
  va_start(va, fmt);
  vsprintf(buf, fmt, va);
  va_end(va);
  fprintf(stderr, "%s\n", buf);
  return 0;
}

static int rerror(int e, char *fmt, ...)
{
  char buf[BUFSIZE];
  va_list va;
  va_start(va, fmt);
  vsprintf(buf, fmt, va);
  va_end(va);
  fprintf(stderr, "%s\n", buf);
  return e;
}

static uint getXPMcolor(XPMINFO *xi, char *c)
{
  int i, len = XPM_COLOR_BUF - 1;
  if(!c || !strncmp("none", c, len) || !strncmp("None", c, len))
    return xi->color_none;
  for(i = 0; i < sizeof(xpmcolors) / sizeof(xpmcolors[0]); ++i){
    if(!strncmp(xpmcolors[i].c, c, len)) return xpmcolors[i].argb;
  }
  if(c[0] == '#'){
    uint argb;
    int n = sscanf(c, "#%x", &argb);
    if(n){
      if(argb & 0xFF000000) return argb;
      else{
        uchar r = (argb >> 16) & 0x0FF;
        uchar g = (argb >> 8) & 0x0FF;
        uchar b = argb & 0x0FF;
#if 0
        uchar rt = 223, gt = 183, bt = 191;
#else
        uchar rt = 223, gt = 187, bt = 191;
#endif
        uchar bg = 8 + ((r>rt ? 4 : 0) | (g>gt ? 2 : 0) | (b>bt ? 1 : 0));
        uchar fg = !bg ? 8 : (bg == 0x0F ? 7 : (0x0F - bg));
        return (((bg << 4) | (fg & 0x0F)) << 24) | argb;
      }
    }
  }
  rerror(1, "unknown color: [%s]", c);
  return xi->color_none;
}

static int setXPMpal(XPMINFO *xi, int n, char *s, char *c, char *m, char *g,
  uint a, char *p)
{
  XPMCOLOR *t = &xi->pal[n];
  XPMNCPY(t->s, s); XPMNCPY(t->c, c); XPMNCPY(t->m, m); XPMNCPY(t->g, g);
  t->argb = a; XPMNCPY(t->p, p);
  return 0;
}

static uint pickXPMpal(XPMINFO *xi, char *p)
{
  int i;
  for(i = 0; i < XPM_PALETTE_MAX; ++i){
    if(!strncmp(xi->pal[i].p, p, xi->cpp)) return xi->pal[i].argb;
  }
  rerror(1, "unknown color palette: [%s]", p);
  return xi->color_none;
}

static int buildBMP(XPMINFO *xi)
{
  int x, y;
  xi->bpp = xi->p * xi->d / 8; // bytes per pixel
  xi->wlen = xi->c * xi->bpp; // bytes per line
  xi->sz = xi->r * xi->wlen; // data bytes
  xi->a = (uint *)malloc(xi->sz); // (uint *) (A)RGB L.E. -> (BYTE *) B,G,R,A
  if(!xi->a) return rerror(1, "cannot allocate pixel buffer");
  // initialize pixel buffer
  for(y = 0; y < xi->r; ++y){
#if 0
    uint t = y & 1 ? 0x000000FF : 0x0000FF00;
    for(x = 0; x < xi->c; ++x) xi->a[xi->c * y + x] = x & 1 ? t : t << 8;
#else
    for(x = 0; x < xi->c; ++x) xi->a[xi->c * y + x] = xi->pal[0].argb;
#endif
  }
  return 0;
}

static int loadXPMINFO(XPMINFO *xi, char *buf)
{
  int r;
  char remain[BUFSIZE];
  if(buf[0] != '"' || buf[strlen(buf) - 2] != '"'
  || buf[strlen(buf) - 1] != ',') return rerror(1, "no XPMINFO: %s", buf);
  r = sscanf(buf, "\"%d %d %d %d %d %d %[^\"]s",
    &xi->c, &xi->r, &xi->ncolors, &xi->cpp, &xi->x_hot, &xi->y_hot, remain);
  if(r < 4) return rerror(1, "XPMINFO requires 4 integers: %s", buf);
  if(r >= 7 && !strncmp(remain, XPM_FMT_XPMEXT, strlen(XPM_FMT_XPMEXT)))
    xi->xpmext = 1;
#if 0
  rprintf("%d[%d][%d][%d][%d][%d][%d][%d]",
    r, xi->c, xi->r, xi->ncolors, xi->cpp, xi->x_hot, xi->y_hot, xi->xpmext);
#endif
  if(xi->ncolors > XPM_PALETTE_MAX)
    return rerror(1, "too much palettes: %d", xi->ncolors);
  if(xi->cpp > XPM_CPP_MAX)
    return rerror(1, "too large cpp: %d", xi->cpp);
  return 0;
}

static int loadXPMpal(XPMINFO *xi, int n, char *buf)
{
  char p[XPM_CPP_MAX + 1] = {0};
  char *s = NULL, *c = NULL, *m = NULL, *g = NULL, *q, *r;
  buf[strlen(buf) - 2] = '\0';
  strncpy(p, buf + 1, xi->cpp); if(p[1] == '\t') p[1] = ' ';
  for(q = buf + 1 + (xi->cpp <= 1 ? 2 : xi->cpp); *q; q = r + 1){
    switch(*q){
    case 's': s = q + 2; break;
    case 'c': c = q + 2; break;
    case 'm': m = q + 2; break;
    case 'g': g = q + 2; break;
    default: rerror(1, "unknown color descriptor: [%s] %s", p, q);
    }
    if(r = strchr(q, '\t')) *r = '\0';
    else break;
  }
  uint argb = getXPMcolor(xi, c);
  if(xi->mode & 0x00000001){
    if(argb == xi->color_none) argb &= 0x00FFFFFF;
    else argb |= 0xFF000000;
  }
  return setXPMpal(xi, n, s, c, m, g, argb, p);
}

static int loadXPMpixel(XPMINFO *xi, int n, char *buf)
{
  int q;
  buf[1 + xi->c * xi->cpp] = '\0';
  for(q = 0; q < xi->c; ++q)
    xi->a[n * xi->c + q] = pickXPMpal(xi, &buf[1 + q * xi->cpp]); // ARGB
  return 0;
}

static int skipcomment(char *buf, int flag) // fake parser
{
  if(!flag){
    char *p = strstr(buf, "//");
    if(p){ *p = '\0'; return 0; }
    p = strstr(buf, "/*");
    if(p){
      char *q = strstr(p, "*/");
      *p = '\0';
      return q ? 0 : 1;
    }
    return 0;
  }else{
    char *q = strstr(buf, "*/");
    *buf = '\0';
    return q ? 0 : 1;
  }
}

static int rstrip(char *buf)
{
  int len = strlen(buf);
  if(!len) return len;
  if(buf[len - 1] == 0x0A) buf[len - 1] = '\0';
  len = strlen(buf);
  if(!len) return len;
  if(buf[len - 1] == 0x0D) buf[len - 1] = '\0';
  return strlen(buf);
}

static char *sgets(char *dst, int num, char **src)
{
  char *p = *src;
  int n = 0;
  while(n < num - 1 && *p){
    int lf = (*p == '\n');
    *dst++ = *p++;
    ++n;
    if(lf) break;
  }
  if(!n) return NULL;
  *dst = '\0';
  *src = p;
  return dst;
}

__PORT uint loadxpm(XPMINFO *xi, char *xpmbuffer)
{
  char *sp = xpmbuffer;
  char buf[BUFSIZE];
  int stat = 0, flag = 0, line = 0;
  if(!xi) return 1;
  memset(xi, 0, sizeof(XPMINFO));
  // test (c=8, r=4, p=4, d=8) 128B=32W(=r4xc8)=32B/line=4B/pixel(=d8p4)
  xi->c = 64, xi->r = 38, xi->p = 4, xi->d = 8;
  xi->color_none = XPM_COLOR_NONE, xi->mode = 1;
  setXPMpal(xi, 0, "none", "none", "none", "none", xi->color_none, XPM_PNONE);
  rprintf("loading: len=%d", strlen(xpmbuffer));
  while(sgets(buf, sizeof(buf) / sizeof(buf[0]), &sp)){
    if(!rstrip(buf)) continue;
    if(!stat){
      if(!strncmp(buf, XPM_FMT_FIRST, strlen(XPM_FMT_FIRST))) ++stat;
      else{ rerror(1, "line 0 expected: %s", XPM_FMT_FIRST); break; }
    }else if(stat == 1){
      if(flag = skipcomment(buf, flag)) continue; // fake parser
      if(!strlen(buf)) continue;
      if(!line){
        if(!strncmp(buf, XPM_FMT_SECOND, strlen(XPM_FMT_SECOND))
        && buf[strlen(buf) - 1] == '{') ++line;
        else{ rerror(1, "unknown statement: %s", buf); break; }
      }else if(line == 1){
        if(loadXPMINFO(xi, buf)) break;
        if(!xi->a){ if(buildBMP(xi)) break; }
        ++line;
      }else if(line >= XPML && line < XPML + xi->ncolors){
        if(loadXPMpal(xi, line++ - XPML, buf)) break;
      }else{
        if(loadXPMpixel(xi, line++ - xi->ncolors - XPML, buf)) break;
      }
    }else{
      rerror(1, "unknown stat: %d", stat);
    }
  }
  if(!xi->a) return 1;
  rprintf("done. (%d, %d, %d): %d", xi->r, xi->c, xi->ncolors, xi->cpp);
  return 0;
}

__PORT uint freexpm(XPMINFO *xi)
{
  if(!xi) return 1;
  if(xi->a){ free(xi->a); xi->a = NULL; }
  return 0;
}

__PORT uint getxpminfosize()
{
  return sizeof(XPMINFO);
}

static int tbInfo(int line, PyCodeObject *f_code)
{
  // struct _frame in frameobject.h
  //   (frame->f_code->..)
  //   (tb->tb_frame->f_code->..)
  char *file = PyString_AsString(f_code->co_filename);
  char *fnc = PyString_AsString(f_code->co_name);
  fprintf(stderr, "    %s(%d): %s\n", file, line, fnc);
  return 0;
}

static int tbDisp(char *s)
{
  fprintf(stderr, "Traceback (most recent call last): --[%s]--\n", s ? s:"");
  PyThreadState *tstat = PyThreadState_GET();
  if(tstat && tstat->frame){
    PyFrameObject *frame = tstat->frame;
    if(!frame) fprintf(stderr, "  error: [!frame] broken stack ?\n");
    else{
      // PyTracebackObject* tb = (PyTracebackObject*)tstat->curexc_traceback;
      // PyTracebackObject* tb = (PyTracebackObject*)tstat->exc_traceback;
      // PyTracebackObject* tb = (PyTracebackObject*)tstat->async_exc;
      PyTracebackObject* tb = (PyTracebackObject*)frame->f_exc_traceback;
      if(!tb){
        fprintf(stderr, "  error: [!tb] another stack ?\n");
        while(frame){ // backword
          // tbInfo(frame->f_lineno, frame->f_code); // not the correct number
          /* need to call PyCode_Addr2Line() */
          tbInfo(PyCode_Addr2Line(frame->f_code, frame->f_lasti), frame->f_code);
          frame = frame->f_back;
        }
      }else{
        while(tb){ // forward
          tbInfo(tb->tb_lineno, tb->tb_frame->f_code); // is tb_lineno correct ?
          tb = tb->tb_next;
        }
      }
    }
  }else{
    fprintf(stderr, "  error: [!tstat || !tstat->frame] another thread ?\n");
  }
  return 0;
}

PyObject *xpmProcessException(PyObject *self, PyObject *args, PyObject *kw)
{
  char *s;
  PyObject *ptyp = NULL;
  PyObject *pval = NULL;
  PyObject *ptb = NULL;

  FILE *fp = fopen(TESTLOG, "ab");
  fprintf(fp, "xpmProcessException %08x %08x %08x\n",
    (uchar *)self, (uchar *)args, (uchar *)kw);
  fclose(fp);

  // if(obj == Py_None){ }

  char *keys[] = {"s", "typ", "val", "tb", NULL};
  if(!PyArg_ParseTupleAndKeywords(args, kw, "|sOOO", keys, &s, &ptyp, &pval, &ptb)){
    FILE *fp = fopen(TESTLOG, "ab");
    fprintf(fp, "ERROR: PyArg_ParseTupleAndKeywords()\n");
    fclose(fp);
    return NULL;
  }else{
    FILE *fp = fopen(TESTLOG, "ab");
    fprintf(fp, "SUCCESS: PyArg_ParseTupleAndKeywords(%s, %08x, %08x, %08x)\n", s, (char *)ptyp, (char *)pval, (char *)ptb);
    fclose(fp);
  }

  tbDisp(s);

  PyObject *mtb = PyImport_ImportModule("traceback");
  if(!mtb) fprintf(stderr, "cannot import traceback\n");
  else{
    char *fmt[] = {"format_exception_only", "format_exception"};
    PyObject *formatted_list;
    if(!ptb) formatted_list = PyObject_CallMethod(mtb, fmt[0],
      "OO", ptyp, pval);
    else formatted_list = PyObject_CallMethod(mtb, fmt[1],
      "OOO", ptyp, pval, ptb);
    if(!formatted_list){
      fprintf(stderr, "None == traceback.%s(...)\n", fmt[ptb ? 1 : 0]);
    }else{
      long len = PyLong_AsLong(
        PyObject_CallMethod(formatted_list, "__len__", NULL));
      if(0) fprintf(stderr, "traceback.%s(...): %d\n", fmt[ptb ? 1 : 0], len);
      long i;
      for(i = 0; i < len; ++i)
        fprintf(stderr, "%s", PyString_AsString(
          PyList_GetItem(formatted_list, i)));
    }
  }
  return Py_BuildValue("{ss}", "s", s);
}

PyObject *XPM(PyObject *self, PyObject *args, PyObject *kw)
{
  char *buf;
  PyObject *nda = NULL; // PyArrayObject *nda = NULL;

#if 0
  FILE *fp = fopen(TESTLOG, "ab");
  fprintf(fp, "XPM %08x %08x %08x\n",
    (uchar *)self, (uchar *)args, (uchar *)kw);
  fclose(fp);
#endif

  char *keys[] = {"buf", NULL};
  if(!PyArg_ParseTupleAndKeywords(args, kw, "|s", keys, &buf)){
    FILE *fp = fopen(TESTLOG, "ab");
    fprintf(fp, "ERROR: PyArg_ParseTupleAndKeywords()\n");
    fclose(fp);
    Py_RETURN_NONE; // must raise Exception
  }else{
#if 0
    FILE *fp = fopen(TESTLOG, "ab");
    fprintf(fp, "SUCCESS: PyArg_ParseTupleAndKeywords(len=%d)\n", strlen(buf));
    fclose(fp);
#endif
  }

  if(buf){
#if 1
    // PyDict_SetItemString(pdi, "buf", PyString_FromString(buf));
    // XPMPROCESSEXCEPTION("XPM");
    XPMINFO xi;
    if(!loadxpm(&xi, buf)){
      npy_intp dims[] = {xi.r, xi.c, 4}; // shapes {rows, cols, colors: RGBA}
      int ndim = sizeof(dims) / sizeof(dims[0]);
      nda = PyArray_SimpleNew(ndim, dims, NPY_UINT8);
      if(nda) memcpy(PyArray_DATA(nda), xi.a, dims[0] * dims[1] * dims[2]);
      freexpm(&xi);
    }
#else
    npy_intp dims[] = {5, 7, 4}; // shapes {rows, cols, colors: RGBA}
    int ndim = sizeof(dims) / sizeof(dims[0]);
#if 0
    static uint dummy[] = { // (uint *) ABGR -> (uchar *) R,G,B,A (dims[2] = 4)
 0xFF00FFFF,0xFF0000FF,0xFF00FFFF,0xFF0000FF,0xFF00FFFF,0xFF0000FF,0xFF00FFFF,
 0xFFFF0000,0xFF00FFFF,0xFFFF0000,0xFF00FFFF,0xFFFF0000,0xFF00FFFF,0xFFFF0000,
 0xFF00FFFF,0xFF00FF00,0xFF00FFFF,0xFF00FF00,0xFF00FFFF,0xFF00FF00,0xFF00FFFF,
 0xFF33AAEE,0xFFFFFF00,0xFF33AAEE,0xFFFF00FF,0xFF33AAEE,0xFF00FFFF,0xFF33AAEE,
 0xFFFFFF00,0xFF33AAEE,0xFFFF00FF,0xFF33AAEE,0xFF00FFFF,0xFF33AAEE,0xCC33AAEE};
#else
    static uint dummy[] = { // (uint *) ARGB -> (uchar *) B,G,R,A (dims[2] = 4)
 0xFFFFFF00,0xFFFF0000,0xFFFFFF00,0xFFFF0000,0xFFFFFF00,0xFFFF0000,0xFFFFFF00,
 0xFF0000FF,0xFFFFFF00,0xFF0000FF,0xFFFFFF00,0xFF0000FF,0xFFFFFF00,0xFF0000FF,
 0xFFFFFF00,0xFF00FF00,0xFFFFFF00,0xFF00FF00,0xFFFFFF00,0xFF00FF00,0xFFFFFF00,
 0xFFEEAA33,0xFF00FFFF,0xFFEEAA33,0xFFFF00FF,0xFFEEAA33,0xFFFFFF00,0xFFEEAA33,
 0xFF00FFFF,0xFFEEAA33,0xFFFF00FF,0xFFEEAA33,0xFFFFFF00,0xFFEEAA33,0xCCEEAA33};
#endif
    nda = PyArray_SimpleNewFromData(ndim, dims, NPY_UINT8, dummy);
#endif
  }
  if(!nda){ // if(nda == Py_None)
    Py_RETURN_NONE; // must raise Exception
  }
  Py_INCREF(nda);
  return nda;
}

PyObject *XPMINFOSIZE(PyObject *self)
{
  return Py_BuildValue("i", getxpminfosize());
}

static PyMethodDef xpm_methods[] = {
  {"xpmProcessException", (PyCFunction)xpmProcessException,
    METH_VARARGS | METH_KEYWORDS, "*args, **kw:\n"
    " s:\n"
    " typ:\n"
    " val:\n"
    " tb:\n"
    "result: dict (output to stderr)"},
  {"XPM", (PyCFunction)XPM,
    METH_VARARGS | METH_KEYWORDS, "*args, **kw:\n"
    " buf: .XPM data buffer\n"
    "result: ndarray"},
  {"XPMINFOSIZE", (PyCFunction)XPMINFOSIZE,
    METH_NOARGS, "no args:\n"
    "result: int"},
  {NULL, NULL, 0, NULL}
};

static char xpm_docstr[] = \
  "about this module\n"\
  "XPM image file loader for Python (to numpy ndarray or PIL) native C .pyd";

PyMODINIT_FUNC initxpm()
{
  PyObject *m = Py_InitModule3(_XPM, xpm_methods, xpm_docstr);
  if(!m) return;
  /* IMPORTANT: this must be called */
  import_array();
}
