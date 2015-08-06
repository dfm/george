#ifndef _AUTODIFF_H_
#define _AUTODIFF_H_

#include <cmath>
#include <cfloat>
#include <iostream>

namespace george {
namespace autodiff {

template <typename T>
struct Jet {
    // Constructors.
    Jet() : a(), v(0.0) {};

    explicit Jet (const T& value) {
        a = value;
        v = T(0.0);
    }

    Jet (const T& value, const T& delta) {
        a = value;
        v = T(delta);
    }

    // Compound operators
    Jet<T>& operator+=(const Jet<T> &y) {
        *this = *this + y;
        return *this;
    }

    Jet<T>& operator-=(const Jet<T> &y) {
        *this = *this - y;
        return *this;
    }

    Jet<T>& operator*=(const Jet<T> &y) {
        *this = *this * y;
        return *this;
    }

    Jet<T>& operator/=(const Jet<T> &y) {
        *this = *this / y;
        return *this;
    }

    T a, v;
};

// Unary +
template<typename T> inline
Jet<T> const& operator+(const Jet<T>& f) {
    return f;
}

// Unary -
template<typename T> inline
Jet<T> operator-(const Jet<T>&f) {
    return Jet<T>(-f.a, -f.v);
}

// Binary +
template<typename T> inline
Jet<T> operator+(const Jet<T>& f,
                 const Jet<T>& g) {
    return Jet<T>(f.a + g.a, f.v + g.v);
}

// Binary + with a scalar: x + s
template<typename T> inline
Jet<T> operator+(const Jet<T>& f, T s) {
    return Jet<T>(f.a + s, f.v);
}

// Binary + with a scalar: s + x
template<typename T> inline
Jet<T> operator+(T s, const Jet<T>& f) {
    return Jet<T>(f.a + s, f.v);
}

// Binary -
template<typename T> inline
Jet<T> operator-(const Jet<T>& f,
                 const Jet<T>& g) {
    return Jet<T>(f.a - g.a, f.v - g.v);
}

// Binary - with a scalar: x - s
template<typename T> inline
Jet<T> operator-(const Jet<T>& f, T s) {
    return Jet<T>(f.a - s, f.v);
}

// Binary - with a scalar: s - x
template<typename T> inline
Jet<T> operator-(T s, const Jet<T>& f) {
    return Jet<T>(s - f.a, -f.v);
}

// Binary *
template<typename T> inline
Jet<T> operator*(const Jet<T>& f,
                 const Jet<T>& g) {
    return Jet<T>(f.a * g.a, f.a * g.v + f.v * g.a);
}

// Binary * with a scalar: x * s
template<typename T> inline
Jet<T> operator*(const Jet<T>& f, T s) {
    return Jet<T>(f.a * s, f.v * s);
}

// Binary * with a scalar: s * x
template<typename T> inline
Jet<T> operator*(T s, const Jet<T>& f) {
    return Jet<T>(f.a * s, f.v * s);
}

// Binary /
template<typename T> inline
Jet<T> operator/(const Jet<T>& f,
        const Jet<T>& g) {
    // This uses:
    //
    //   a + u   (a + u)(b - v)   (a + u)(b - v)
    //   ----- = -------------- = --------------
    //   b + v   (b + v)(b - v)        b^2
    //
    // which holds because v*v = 0.
    const T g_a_inverse = T(1.0) / g.a;
    const T f_a_by_g_a = f.a * g_a_inverse;
    return Jet<T>(f.a * g_a_inverse, (f.v - f_a_by_g_a * g.v) * g_a_inverse);
}

// Binary / with a scalar: s / x
template<typename T> inline
Jet<T> operator/(T s, const Jet<T>& g) {
    const T minus_s_g_a_inverse2 = -s / (g.a * g.a);
    return Jet<T>(s / g.a, g.v * minus_s_g_a_inverse2);
}

// Binary / with a scalar: x / s
template<typename T> inline
Jet<T> operator/(const Jet<T>& f, T s) {
    const T s_inverse = 1.0 / s;
    return Jet<T>(f.a * s_inverse, f.v * s_inverse);
}

// Binary comparison operators for both scalars and jets.
#define DEFINE_JET_COMPARISON_OPERATOR(op) \
template<typename T> inline \
bool operator op(const Jet<T>& f, const Jet<T>& g) { \
    return f.a op g.a; \
} \
template<typename T> inline \
bool operator op(const T& s, const Jet<T>& g) { \
    return s op g.a; \
} \
template<typename T> inline \
bool operator op(const Jet<T>& f, const T& s) { \
    return f.a op s; \
}
DEFINE_JET_COMPARISON_OPERATOR( <  )
DEFINE_JET_COMPARISON_OPERATOR( <= )
DEFINE_JET_COMPARISON_OPERATOR( >  )
DEFINE_JET_COMPARISON_OPERATOR( >= )
DEFINE_JET_COMPARISON_OPERATOR( == )
DEFINE_JET_COMPARISON_OPERATOR( != )
#undef DEFINE_JET_COMPARISON_OPERATOR

// Pull some functions from namespace std.
//
// This is necessary because we want to use the same name (e.g. 'sqrt') for
// double-valued and Jet-valued functions, but we are not allowed to put
// Jet-valued functions inside namespace std.
//
inline double abs     (double x) { return std::abs(x);      }
inline double log     (double x) { return std::log(x);      }
inline double exp     (double x) { return std::exp(x);      }
inline double sqrt    (double x) { return std::sqrt(x);     }
inline double cos     (double x) { return std::cos(x);      }
inline double acos    (double x) { return std::acos(x);     }
inline double sin     (double x) { return std::sin(x);      }
inline double asin    (double x) { return std::asin(x);     }
inline double tan     (double x) { return std::tan(x);      }
inline double atan    (double x) { return std::atan(x);     }
inline double sinh    (double x) { return std::sinh(x);     }
inline double cosh    (double x) { return std::cosh(x);     }
inline double tanh    (double x) { return std::tanh(x);     }
inline double pow  (double x, double y) { return std::pow(x, y);   }
inline double atan2(double y, double x) { return std::atan2(y, x); }

// In general, f(a + h) ~= f(a) + f'(a) h, via the chain rule.

// abs(x + h) ~= x + h or -(x + h)
template <typename T> inline
Jet<T> abs(const Jet<T>& f) {
    return f.a < T(0.0) ? -f : f;
}

// log(a + h) ~= log(a) + h / a
template <typename T> inline
Jet<T> log(const Jet<T>& f) {
    const T a_inverse = T(1.0) / f.a;
    return Jet<T>(log(f.a), f.v * a_inverse);
}

// exp(a + h) ~= exp(a) + exp(a) h
template <typename T> inline
Jet<T> exp(const Jet<T>& f) {
    const T tmp = exp(f.a);
    return Jet<T>(tmp, tmp * f.v);
}

// sqrt(a + h) ~= sqrt(a) + h / (2 sqrt(a))
template <typename T> inline
Jet<T> sqrt(const Jet<T>& f) {
    T tmp = sqrt(f.a);
    const T two_a_inverse = T(1.0) / (T(2.0) * tmp);
    return Jet<T>(tmp, f.v * two_a_inverse);
}

// cos(a + h) ~= cos(a) - sin(a) h
template <typename T> inline
Jet<T> cos(const Jet<T>& f) {
    return Jet<T>(cos(f.a), - sin(f.a) * f.v);
}

// acos(a + h) ~= acos(a) - 1 / sqrt(1 - a^2) h
template <typename T> inline
Jet<T> acos(const Jet<T>& f) {
    const T tmp = - T(1.0) / sqrt(T(1.0) - f.a * f.a);
    return Jet<T>(acos(f.a), tmp * f.v);
}

// sin(a + h) ~= sin(a) + cos(a) h
template <typename T> inline
Jet<T> sin(const Jet<T>& f) {
    return Jet<T>(sin(f.a), cos(f.a) * f.v);
}

// asin(a + h) ~= asin(a) + 1 / sqrt(1 - a^2) h
template <typename T> inline
Jet<T> asin(const Jet<T>& f) {
    const T tmp = T(1.0) / sqrt(T(1.0) - f.a * f.a);
    return Jet<T>(asin(f.a), tmp * f.v);
}

// tan(a + h) ~= tan(a) + (1 + tan(a)^2) h
template <typename T> inline
Jet<T> tan(const Jet<T>& f) {
    const T tan_a = tan(f.a);
    const T tmp = T(1.0) + tan_a * tan_a;
    return Jet<T>(tan_a, tmp * f.v);
}

// atan(a + h) ~= atan(a) + 1 / (1 + a^2) h
template <typename T> inline
Jet<T> atan(const Jet<T>& f) {
    const T tmp = T(1.0) / (T(1.0) + f.a * f.a);
    return Jet<T>(atan(f.a), tmp * f.v);
}

// sinh(a + h) ~= sinh(a) + cosh(a) h
template <typename T> inline
Jet<T> sinh(const Jet<T>& f) {
    return Jet<T>(sinh(f.a), cosh(f.a) * f.v);
}

// cosh(a + h) ~= cosh(a) + sinh(a) h
template <typename T> inline
Jet<T> cosh(const Jet<T>& f) {
    return Jet<T>(cosh(f.a), sinh(f.a) * f.v);
}

// tanh(a + h) ~= tanh(a) + (1 - tanh(a)^2) h
template <typename T> inline
Jet<T> tanh(const Jet<T>& f) {
    const T tanh_a = tanh(f.a);
    const T tmp = T(1.0) - tanh_a * tanh_a;
    return Jet<T>(tanh_a, tmp * f.v);
}

// atan2(b + db, a + da) ~= atan2(b, a) + (- b da + a db) / (a^2 + b^2)
//
// In words: the rate of change of theta is 1/r times the rate of
// change of (x, y) in the positive angular direction.
template <typename T> inline
Jet<T> atan2(const Jet<T>& g, const Jet<T>& f) {
    // Note order of arguments:
    //
    //   f = a + da
    //   g = b + db

    T const tmp = T(1.0) / (f.a * f.a + g.a * g.a);
    return Jet<T>(atan2(g.a, f.a), tmp * (- g.a * f.v + f.a * g.v));
}


// pow -- base is a differentiable function, exponent is a constant.
// (a+da)^p ~= a^p + p*a^(p-1) da
template <typename T> inline
Jet<T> pow(const Jet<T>& f, double g) {
    T const tmp = g * pow(f.a, g - T(1.0));
    return Jet<T>(pow(f.a, g), tmp * f.v);
}

// pow -- base is a constant, exponent is a differentiable function.
// (a)^(p+dp) ~= a^p + a^p log(a) dp
template <typename T> inline
Jet<T> pow(double f, const Jet<T>& g) {
    T const tmp = pow(f, g.a);
    return Jet<T>(tmp, log(f) * tmp * g.v);
}


// pow -- both base and exponent are differentiable functions.
// (a+da)^(b+db) ~= a^b + b * a^(b-1) da + a^b log(a) * db
template <typename T> inline
Jet<T> pow(const Jet<T>& f, const Jet<T>& g) {
  T const tmp1 = pow(f.a, g.a);
  T const tmp2 = g.a * pow(f.a, g.a - T(1.0));
  T const tmp3 = tmp1 * log(f.a);

  return Jet<T>(tmp1, tmp2 * f.v + tmp3 * g.v);
}

template <typename T>
inline std::ostream &operator<<(std::ostream &s, const Jet<T>& z) {
    return s << "[" << z.a << " ; " << z.v << "]";
}

}; // namespace autodiff
}; // namespace george

#endif
