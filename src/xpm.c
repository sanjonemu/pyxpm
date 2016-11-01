/*
  xpm.c

  >mingw32-make -f makefile.tdmgcc64
  >test_xpm.py
*/

#include <xpm.h>

#include <numpy/arrayobject.h>

#define DEBUGLOG 0
#define TESTLOG "../dll/_test_dll_.log"

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
  char *fn;
// must be ndarray
  PyObject *pdi = PyDict_New();

  // PyObject *np = PyImport_ImportModule("numpy");

  FILE *fp = fopen(TESTLOG, "ab");
  fprintf(fp, "XPM %08x %08x %08x\n",
    (uchar *)self, (uchar *)args, (uchar *)kw);
  fclose(fp);

  // if(obj == Py_None){ }

  char *keys[] = {"fn", NULL};
  if(!PyArg_ParseTupleAndKeywords(args, kw, "|s", keys, &fn)){
    FILE *fp = fopen(TESTLOG, "ab");
    fprintf(fp, "ERROR: PyArg_ParseTupleAndKeywords()\n");
    fclose(fp);
    return NULL;
  }else{
    FILE *fp = fopen(TESTLOG, "ab");
    fprintf(fp, "SUCCESS: PyArg_ParseTupleAndKeywords(%s)\n", fn);
    fclose(fp);
  }

// must be ndarray
  if(fn) PyDict_SetItemString(pdi, "fn", PyString_FromString(fn));
  // XPMPROCESSEXCEPTION("XPM");

// must be ndarray
  return Py_BuildValue("{sO}", "o", pdi);
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
    " fn: .XPM file name\n"
    "result: ndarray"},
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
