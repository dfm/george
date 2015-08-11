.. _modeling:

Modeling language
=================

In order to make hyperparameter optimization, george comes with a modeling
language specification. To this end, any component of the model that is
specified by parameters that we might want to fit should conform to a simple
modeling protocol. Each element should expose a value, a parameter vector, and
the gradient of the value with respect to these parameters. Some places where
this comes in handy is for the hyperparameters of the kernels and the metrics.

The protocol
------------

We don't really care too much about the implementation details of your objects
but if you want them to satisfy this protocol, they must satisfy all of the
following methods:

.. code-block:: python

    import numpy as np

    class ModelingProtocol(object):

        def __len__(self):
            """
            Return the number of (un-frozen) parameters exposed by this
            object.

            """
            return 10

        def get_parameter_names(self):
            """
            Return a list with the names of the parameters.

            """
            return ["param1", "param2", ...]

        def get_vector(self):
            """
            Returns a numpy array with the current settings of all the
            non-frozen parameters.

            """
            return np.array([self.par1, self.par2, ...])

        def set_vector(self, vector):
            """
            Update the vector with a numpy array of the same shape and order
            as ``get_vector``.

            """
            self.par1 = vector[0]
            self.par2 = vector[1]
            ...

        def get_value(self, *args):
            """
            Get the value of this object at the current setting of the
            parameters. This method can optionally take arguments. For a
            kernel, these extra arguments would be the input coordinates.

            """
            return f(*args)

        def get_gradient(self, *args):
            """
            Get the gradient of ``get_value`` with respect to the parameter
            vector returned by ``get_vector``.

            """
            return dfdx(*args)

        def freeze_parameter(self, parameter_name):
            """
            Fix the value of some parameter by name at its current value. This
            parameter should no longer be returned by ``get_vector`` or
            ``get_gradient``.

            """
            freeze(parameter_name)

        def thaw_parameter(self, parameter_name):
            """
            The opposite of ``freeze_parameter``.

            """
            thaw(parameter_name)


A simple example
----------------

Let's start with a simple example. Let's say that we want to implement a
one-dimensional exponential-squared kernel

.. math::

    k(x_1, \, x_2) = a\,\exp\left(-\frac{(x_1-x_2)^2}{2\,l}\right)

in pure-Python and fit for the logarithms of the parameters. In that case, the
implementation would be something like the following:

.. code-block:: python

    import numpy as np

    class MyNewExpSquared(object):

        def __init__(self, a, l):
            self.parameter_names = ["lna", "lnl"]
            self.parameters = np.array([a, l])
            self.unfrozen = np.ones_like(self.parameters, dtype=bool)

        def __len__(self):
            return np.sum(self.unfrozen)

        def get_parameter_names(self):
            return [n for i, n in enumerate(self.parameter_names)
                    if self.unfrozen[i]]

        def get_vector(self):
            return np.log(self.parameters[self.unfrozen])

        def set_vector(self, vector):
            self.parameters[self.unfrozen] = np.exp(vector)

        def get_value(self, x1, x2):
            a, l = self.parameters
            r2 = (x1 - x2)**2
            return a * np.exp(-0.5 * r2 / l)

        def get_gradient(self, x1, x2):
            a, l = self.parameters
            value = self.get_value(x1, x2)
            grad = np.array((
                value,
                value * (0.5 * (x1 - x2)**2 / l)
            ))
            return grad[self.unfrozen]

        def freeze_parameter(self, parameter_name):
            names = self.parameter_names
            self.unfrozen[names.index(parameter_name)] = False

        def thaw_parameter(self, parameter_name):
            names = self.parameter_names
            self.unfrozen[names.index(parameter_name)] = True
