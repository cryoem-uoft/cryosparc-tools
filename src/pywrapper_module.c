#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL SHARED_ARRAY_ARRAY_API

#include <Python.h>
#include <numpy/arrayobject.h>

#if !defined(MODULENAME)
#define MODULENAME example
#endif

#define XCONCAT(a,b) a ## b
#define CONCAT(a,b) XCONCAT(a,b)
#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

/*
	EDIT THIS TO ADD YOUR FUNCTIONS
	VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
*/

#include "pywrapper_wrapperfunctions.c"

static PyMethodDef module_functions[] = {
	{  "dset_new",            (PyCFunction)  wrap_dset_new,            METH_VARARGS  |  METH_KEYWORDS,  ""  },
	{  "dset_del",            (PyCFunction)  wrap_dset_del,            METH_VARARGS  |  METH_KEYWORDS,  ""  },
	{  "dset_copy",           (PyCFunction)  wrap_dset_del,            METH_VARARGS  |  METH_KEYWORDS,  ""  },
	{  "dset_totalsz",        (PyCFunction)  wrap_dset_totalsz,        METH_VARARGS  |  METH_KEYWORDS,  ""  },
	{  "dset_ncol",           (PyCFunction)  wrap_dset_ncol,           METH_VARARGS  |  METH_KEYWORDS,  ""  },
	{  "dset_nrow",           (PyCFunction)  wrap_dset_nrow,           METH_VARARGS  |  METH_KEYWORDS,  ""  },
	{  "dset_type",           (PyCFunction)  wrap_dset_type,           METH_VARARGS  |  METH_KEYWORDS,  ""  },
	{  "dset_get",            (PyCFunction)  wrap_dset_get,            METH_VARARGS  |  METH_KEYWORDS,  ""  },
	{  "dset_setstr",         (PyCFunction)  wrap_dset_setstr,         METH_VARARGS  |  METH_KEYWORDS,  ""  },
	{  "dset_getstr",         (PyCFunction)  wrap_dset_getstr,         METH_VARARGS  |  METH_KEYWORDS,  ""  },
	{  "dset_addrows",        (PyCFunction)  wrap_dset_addrows,        METH_VARARGS  |  METH_KEYWORDS,  ""  },
	{  "dset_addcol_scalar",  (PyCFunction)  wrap_dset_addcol_scalar,  METH_VARARGS  |  METH_KEYWORDS,  ""  },
	{  "dset_addcol_array",   (PyCFunction)  wrap_dset_addcol_array,   METH_VARARGS  |  METH_KEYWORDS,  ""  },
	{  "dset_defrag",         (PyCFunction)  wrap_dset_defrag,         METH_VARARGS  |  METH_KEYWORDS,  ""  },
	{  "dset_dumptxt",        (PyCFunction)  wrap_dset_dumptxt,        METH_VARARGS  |  METH_KEYWORDS,  ""  },

	{ NULL, NULL, 0, NULL }
};

/*
	^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	===============================
*/



static const char module_name[] = STRINGIFY(MODULENAME);

static struct PyModuleDef module_def = {
	PyModuleDef_HEAD_INIT,
	module_name,      /* m_name */
	NULL,             /* m_doc */
	-1,               /* m_size */
	module_functions, /* m_methods */
	NULL,             /* m_reload */
	NULL,             /* m_traverse */
	NULL,             /* m_clear  */
	NULL,             /* m_free */
};

static PyObject *module_init(void)
{
	PyObject *m;

	// Import numpy arrays
	import_array1(NULL);

	// Register the module
	if (!(m = PyModule_Create(&module_def)))
		return NULL;

	return m;
}

PyMODINIT_FUNC CONCAT(PyInit_,MODULENAME) (void)
{
	return module_init();
}
