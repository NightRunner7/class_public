#ifndef __PARSER__
#define __PARSER__

#include "common.h"

#define _LINE_LENGTH_MAX_ 200 /**< size of the string read in each line of the file (extra characters not taken into account) */
#define _ARGUMENT_LENGTH_MAX_ 30 /**< maximum size of each argument (name or value), including the final null character */

typedef char FileArg[_ARGUMENT_LENGTH_MAX_];

/* after reading a given file, all relevant information stored in this structure, in view of being processed later*/
struct file_content {
  char * filename;
  int size;
  FileArg * name;  /**< list of (size) names */
  FileArg * value; /**< list of (size) values */
  short * read;    /**< set to _TRUE_ if this parameter is effectively read */
};

/**************************************************************/

/*
 * Boilerplate for C++ 
 */
#ifdef __cplusplus
extern "C" {
#endif

int parser_read_file(
		     char * filename,
		     struct file_content * pfc,
		     ErrorMsg errmsg
		     );

int parser_free(
		struct file_content * pfc
		);

int parser_read_line(
		char * line,
		int * is_data,
		char * name,
		char * value,
		ErrorMsg errmsg
		);

int parser_read_int(
		    struct file_content * pfc,
		    char * name,
		    int * value,
		    ErrorMsg errmsg
		    );

int parser_read_double(
		    struct file_content * pfc,
		    char * name,
		    double * value,
		    ErrorMsg errmsg
		    );

int parser_read_string(
		       struct file_content * pfc,
		       char * name,
		       FileArg * value,
		       ErrorMsg errmsg
		       );

int parser_cat(
	       struct file_content * pfc1,
	       struct file_content * pfc2,
	       struct file_content * pfc3,
	       ErrorMsg errmsg
	       );

#ifdef __cplusplus
}
#endif

#endif
