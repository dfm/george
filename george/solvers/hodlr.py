# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["HODLRSolver"]

import numpy as np
from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve

from .basic import BasicSolver
# from ._hodlr import HODLRSolver as HODLRSolverInterface


class HODLRSolver(BasicSolver):

    def __init__(self, kernel, min_size=100, tol=0.1, seed=42):
        self.min_size = min_size
        self.tol = tol
        self.seed = seed
        super(HODLRSolver, self).__init__(kernel)

    def compute(self, x, yerr):
        random = np.random.RandomState(self.seed)
        self.solver = Node(yerr**2, self.kernel.get_value, x, 0, len(x),
                           self.min_size, self.tol, random)
        self.solver.compute()
        self._log_det = self.solver.log_det
        self.computed = True

    def apply_inverse(self, y, in_place=False):
        return self.solver.solve(y, in_place=in_place)

    def dot_solve(self, y):
        return self.solver.dot_solve(y)

    def apply_sqrt(self, r):
        raise NotImplementedError("apply_sqrt is not implemented for the "
                                  "HODLRSolver")

    def get_inverse(self):
        return self.solver.solve(np.eye(self.solver.size), in_place=True)


class Node(object):

    def __init__(self, diag, kernel, x, start, size, min_size, tol,
                 random=None, direction=0, parent=None):
        if random is None:
            random = np.random
        self.diag = diag
        self.kernel = kernel
        self.x = x
        self.start = start
        self.size = size
        self.parent = parent
        self.direction = direction

        half = size // 2
        if half >= min_size:
            self.is_leaf = False

            U, V = low_rank_approx(kernel, x, x, start+half, size-half, start,
                                   half, tol, random)
            self.U = [V, U]
            self.V = [np.array(V), np.array(U)]
            self.rank = U.shape[1]

            self.children = [
                Node(diag, kernel, x, start, half, min_size, tol,
                     random, 0, self),
                Node(diag, kernel, x, start+half, size-half, min_size, tol,
                     random, 1, self),
            ]

        else:
            self.is_leaf = True
            self.children = []

    def to_dense(self):
        s = slice(self.start, self.start+self.size)
        K = self.kernel(self.x[s], self.x[s])
        K[np.diag_indices_from(K)] += self.diag[s]
        return K

    def compute(self):
        self.log_det = 0.0
        for c in self.children:
            c.compute()
        for c in self.children:
            self.log_det += c.log_det

        self.factorize()

        f = 1.0 + self.is_leaf
        self.log_det += f*np.sum(np.log(np.abs(np.diag(self.factor[0]))))

        node = self.parent
        start = self.start
        ind = self.direction
        while node is not None:
            node.U[ind] = self.apply_inverse(node.U[ind], start)
            start = node.start
            ind = node.direction
            node = node.parent

    def factorize(self):
        if self.is_leaf:
            self.factor = cho_factor(self.to_dense(), overwrite_a=True)
        else:
            rank = self.rank
            S = np.eye(2*rank)
            S[:rank, rank:] = np.dot(self.V[1].T, self.U[1])
            S[rank:, :rank] = np.dot(self.V[0].T, self.U[0])
            self.factor = lu_factor(S, overwrite_a=True)

    def apply_inverse(self, x, start):
        start = self.start - start
        if self.is_leaf:
            s = slice(start, start+self.size)
            x[s] = cho_solve(self.factor, x[s], overwrite_b=True)
            return x

        rank = self.rank
        s1 = self.size // 2
        s2 = self.size - s1
        shape = list(x.shape)
        shape[0] = 2*rank
        tmp = np.empty(shape)
        tmp[:rank] = np.dot(self.V[1].T, x[start+s1:start+s1+s2])
        tmp[rank:] = np.dot(self.V[0].T, x[start:start+s1])
        tmp = lu_solve(self.factor, tmp, overwrite_b=True)

        x[start:start+s1] -= np.dot(self.U[0], tmp[:rank])
        x[start+s1:start+s1+s2] -= np.dot(self.U[1], tmp[rank:])
        return x

    def solve(self, x_in, in_place=False):
        # if len(x_in.shape) == 1:
        #     x_in = x_in[:, None]

        if in_place:
            x = x_in
        else:
            x = np.array(x_in)

        for c in self.children:
            c.solve(x, in_place=True)
        self.apply_inverse(x, 0)
        return x

    def dot_solve(self, x):
        b = self.solve(x, in_place=False)
        return np.dot(x.T, b)


def low_rank_approx(kernel, x1, x2, start_row, n_rows, start_col, n_cols, tol,
                    random=None):
    max_rank = max(n_cols, n_rows)
    U = np.empty((n_rows, max_rank))
    V = np.empty((n_cols, max_rank))

    rank = 0
    index = np.arange(n_rows)
    norm = 0.0
    finished = False
    rows = slice(start_row, start_row+n_rows)
    cols = slice(start_col, start_col+n_cols)

    while True:
        while True:
            if not len(index):
                finished = True
                break
            k = random.randint(len(index))
            i = index[k]
            index[k] = index[-1]
            index = index[:-1]

            row = kernel(x1[start_row+i:start_row+i+1], x2[cols])[0]
            row -= np.dot(U[i, :rank], V[:, :rank].T)
            j = np.argmax(np.abs(row))
            if np.abs(row[j]) > 1e-14:
                break

        if finished:
            if not rank:
                K = kernel(x1[rows], x2[cols])
                if n_cols <= n_rows:
                    return K, np.eye(n_cols)
                return np.eye(n_rows), K
            break

        row /= row[j]
        V[:, rank] = row

        col = kernel(x1[rows], x2[start_col+j:start_col+j+1])[:, 0]
        col -= np.dot(U[:, :rank], V[j, :rank])
        U[:, rank] = col

        rank += 1
        if rank >= max_rank:
            break

        rowcol_norm = np.dot(row, row) * np.dot(col, col)
        if rowcol_norm < tol**2 * norm:
            break

        norm += rowcol_norm
        s = np.dot(U[:, :rank-1].T, col)
        norm += 2 * np.sum(np.abs(s))
        s = np.dot(V[:, :rank-1].T, row)
        norm += 2 * np.sum(np.abs(s))

    return U[:, :rank], V[:, :rank]
