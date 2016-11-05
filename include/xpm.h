/*
  xpm.h
*/

#ifndef __XPM_H__
#define __XPM_H__

#ifndef UNICODE
#define UNICODE
#endif

#include <Python.h>
#include <structmember.h>
#include <frameobject.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#ifdef __XPM_MAKE_DLL_
#define __PORT __declspec(dllexport) // make dll mode
#else
#define __PORT __declspec(dllimport) // use dll mode
#endif

typedef enum {
  black, dblue, dgreen, dcyan, dred, dmagenta, dyellow, gray,
  dgray, blue, green, cyan, red, magenta, yellow, white,
} color;

typedef unsigned char uchar;
typedef unsigned int uint;

#define BUFSIZE 4096

#define CON_BG dblue
#define CON_FG cyan

#define XPML 2
#define XPM_FMT_FIRST "/* XPM"
#define XPM_FMT_SECOND "static char *"
#define XPM_FMT_XPMEXT "XPMEXT"
#define XPM_PNONE "          "
#define XPM_CPP_MAX 10 // length of XPM_PNONE (set <= 10)
#define XPM_COLOR_NONE ((((CON_BG << 4) | (CON_FG & 0x0F)) << 24) | 0x00EEAA33)
#define XPM_COLOR_BUF 16 // string length of a descriptor
#define XPM_PALETTE_MAX 256

#define XPMNCPY(D, S) do{ \
  if(S){ strncpy(D, S, sizeof(D) - 1); D[sizeof(D) - 1] = '\0'; }\
  else D[0] = '\0'; \
}while(0)

typedef struct _XPMCOLORMAP {
  uint argb;
  char *c;
} XPMCOLORMAP;

typedef struct _XPMCOLOR {
  char s[XPM_COLOR_BUF]; // str
  char c[XPM_COLOR_BUF]; // color
  char m[XPM_COLOR_BUF]; // mono
  char g[XPM_COLOR_BUF]; // gray
  uint argb; // (DWORD *) (A)RGB L.E. -> (BYTE *) B,G,R,A (PNONE-A is BGFG)
  char p[XPM_CPP_MAX + 1]; // characters per pixel and space for '\0'
  char reserved[1]; // alignment
} XPMCOLOR;

typedef struct _XPMINFO {
  int c, r, p, d; // cols rows planes depth-bits
  int bpp, wlen, sz, xpmext, ncolors, cpp, x_hot, y_hot;
  XPMCOLOR pal[XPM_PALETTE_MAX]; // color table (none = pal[0])
  uint *a; // pixel buffer
  uint *reserved;
  uint color_none;
  uint mode; // alpha value mode = 0: console, 1: ndarray
} XPMINFO;

__PORT uint loadxpm(XPMINFO *xi, char *xpmbuffer);
__PORT uint freexpm(XPMINFO *xi);
__PORT uint getxpminfosize();

#define _XPM "xpm"

// PyErr_Fetch should be called at the same (stack) layer as MACRO placed on.
// and *MUST* be called PyImport_ImportModule etc *AFTER* PyErr_Fetch
#define XPMPROCESSEXCEPTION(S) do{ \
  if(PyErr_Occurred()){ \
    PyObject *ptyp, *pval, *ptb; \
    PyErr_Fetch(&ptyp, &pval, &ptb); \
    if(0) fprintf(stderr, "%08x %08x: %s\n", ptb, pval, \
      pval ? PyString_AsString(pval) : "!pval"); \
    PyObject *m = PyImport_ImportModule(_XPM); \
    if(!m) fprintf(stderr, "cannot import %s\n", _XPM); \
    else{ \
      PyObject *tpl = Py_BuildValue("(s)", S); \
      PyObject *kw = PyDict_New(); \
      if(ptyp) PyDict_SetItemString(kw, "typ", ptyp); \
      if(pval) PyDict_SetItemString(kw, "val", pval); \
      if(ptb) PyDict_SetItemString(kw, "tb", ptb); \
      PyObject_Call(PyObject_GetAttrString(m, "xpmProcessException"), \
        tpl, kw); \
    } \
    PyErr_NormalizeException(&ptyp, &pval, &ptb); \
    PyErr_Clear(); \
    if(0) fprintf(stderr, "cleanup exceptions inside: %s\n", S); \
  } \
}while(0)

PyObject *xpmProcessException(PyObject *self, PyObject *args, PyObject *kw);
PyObject *XPM(PyObject *self, PyObject *args, PyObject *kw);
PyObject *XPMINFOSIZE(PyObject *self);

#endif // __XPM_H__
