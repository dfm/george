#ifndef _GEORGE_SUBSPACE_H_
#define _GEORGE_SUBSPACE_H_

#include <cmath>
#include <vector>

using std::vector;

namespace george {
namespace subspace {

class Subspace {
public:
    Subspace (unsigned ndim, unsigned naxes)
        : ndim_(ndim), naxes_(naxes), axes_(naxes) {};

    unsigned get_ndim () const { return ndim_; };
    unsigned get_naxes () const { return naxes_; };
    unsigned get_axis (const unsigned i) const { return axes_[i]; };
    void set_axis (const unsigned i, const unsigned value) { axes_[i] = value; };

private:
    unsigned ndim_, naxes_;
    vector<unsigned> axes_;
};

}; // namespace subspace
}; // namespace george

#endif
