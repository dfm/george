#ifndef _GEORGE_SUBSPACE_H_
#define _GEORGE_SUBSPACE_H_

#include <cmath>
#include <vector>

namespace george {
  namespace subspace {

    class Subspace {
      public:
        Subspace (size_t ndim, size_t naxes)
          : ndim_(ndim)
          , naxes_(naxes)
          , axes_(naxes) {};

        size_t get_ndim () const { return ndim_; };
        size_t get_naxes () const { return naxes_; };
        size_t get_axis (size_t i) const { return axes_[i]; };
        void set_axis (size_t i, size_t value) { axes_[i] = value; };

      private:
        size_t ndim_, naxes_;
        std::vector<size_t> axes_;
    };

  }; // namespace subspace
}; // namespace george

#endif
