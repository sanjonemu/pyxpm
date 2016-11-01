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

#ifdef __XPM_MAKE_DLL_
#define __PORT __declspec(dllexport) // make dll mode
#else
#define __PORT __declspec(dllimport) // use dll mode
#endif

typedef unsigned char uchar;
typedef unsigned int uint;

typedef struct _XPMINFO {
  uint *a; // pixel buffer
} XPMINFO;

__PORT uint loadxpm(XPMINFO *xi, char *xpmfile);

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

#endif // __XPM_H__
