#ifndef _GEORGE_EXCEPTIONS_H_
#define _GEORGE_EXCEPTIONS_H_

#include <exception>

namespace george {

struct dimension_mismatch : public std::exception {
  const char * what () const throw () {
    return "dimension mismatch";
  }
};

}

#endif
