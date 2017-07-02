.. module:: george.modeling

.. _modeling-protocol:

Modeling Protocol
=================

This module provides some infrastructure that makes it easy to implement
abstract "models" to be used within the george framework. Many of the
methods are probably more generally applicable but the implementation
constraints can be simplified since we're just concerned about supporting the
needs of george.

The basic premise is that a :class:`Model` is an object that has an ordered set
of named parameters. These parameters are assumed to be continuous but they can
have bounds. There is also the concept of an "active set" of parameters that
are being varied in a fit procedure. The other parameters are "frozen" to a
particular value. Frozen parameters can be "thawed" to be returned to the
active set.

There isn't a formal requirement for the "value" interface that a
:class:`Model` subclass should implement but in some cases, a model will be
expected to implement a ``get_value`` method that returns the "value" of the
model (this can mean many different things but we'll motivate this with an
example below) for the current setting of the parameters.

Since these models will be used in the context of Bayesian parameter estimation
each model also implements a :func:`Model.log_prior` method that computes the
log of the prior probability of the current setting of the model parameters.

The full interface is described in detail below and the tutorials demonstrate
the basic usage of the protocol.

.. autoclass:: george.modeling.Model
   :inherited-members:

.. autoclass:: george.modeling.ModelSet
   :members:

.. autoclass:: george.modeling.ConstantModel
   :members:

.. autoclass:: george.modeling.CallableModel
   :members:
