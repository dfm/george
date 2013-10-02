#include "cholmod.h"

int main ()
{
#ifdef CHOLMOD_HAS_VERSION_FUNCTION
    int version[3];
    cholmod_version (version);
    if (version[0] != 2 || version[1] < 1)
        return -2;
    return 0;
#else
    return -1;
#endif
}
