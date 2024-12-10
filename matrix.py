import numpy as np

class MatrixOperations:
    # Basic Matrix Operations
    @staticmethod
    def add_matrices(*matrices):
        """Add multiple matrices element-wise."""
        return sum(matrices)

    @staticmethod
    def subtract_matrices(*matrices):
        """Subtract multiple matrices element-wise."""
        return matrices[0] - sum(matrices[1:])

    @staticmethod
    def scalar_multiply(matrix, scalar):
        """Multiply every element of the matrix by a scalar."""
        return matrix * scalar

    @staticmethod
    def matrix_multiply(*matrices):
        """Perform dot product (matrix multiplication) for multiple matrices."""
        result = matrices[0]
        for matrix in matrices[1:]:
            result = result @ matrix
        return result

    @staticmethod
    def elementwise_multiply(*matrices):
        """Multiply corresponding elements of multiple matrices (Hadamard product)."""
        result = matrices[0]
        for matrix in matrices[1:]:
            result = result * matrix
        return result

    # Matrix Transformations
    @staticmethod
    def transpose(matrix):
        """Return the transpose of the matrix."""
        return matrix.T

    @staticmethod
    def conjugate_transpose(matrix):
        """Return the conjugate transpose (Hermitian transpose) of the matrix."""
        return np.conjugate(matrix.T)

    @staticmethod
    def reshape(matrix, shape):
        """Reshape the matrix to the given shape."""
        return np.reshape(matrix, shape)

    @staticmethod
    def flatten(matrix):
        """Flatten the matrix into a one-dimensional array."""
        return matrix.flatten()

    # Determinants and Inverses
    @staticmethod
    def determinant(matrix):
        """Compute the determinant of the matrix."""
        return np.linalg.det(matrix)

    @staticmethod
    def inverse(matrix):
        """Compute the inverse of the matrix."""
        return np.linalg.inv(matrix)

    @staticmethod
    def pseudo_inverse(matrix):
        """Compute the pseudo-inverse of the matrix."""
        return np.linalg.pinv(matrix)

    # Special Matrix Functions
    @staticmethod
    def trace(matrix):
        """Compute the trace (sum of diagonal elements) of the matrix."""
        return np.trace(matrix)

    @staticmethod
    def rank(matrix):
        """Compute the rank (number of linearly independent rows or columns) of the matrix."""
        return np.linalg.matrix_rank(matrix)

    @staticmethod
    def norm(matrix, ord=None):
        """Compute the norm (length or size) of the matrix."""
        return np.linalg.norm(matrix, ord=ord)

    @staticmethod
    def condition_number(matrix):
        """Compute the condition number of the matrix."""
        return np.linalg.cond(matrix)

    # Solving Linear Systems
    @staticmethod
    def solve_linear_system(A, b):
        """Solve a linear system of equations Ax = b."""
        return np.linalg.solve(A, b)

    # Eigenvalues and Eigenvectors
    @staticmethod
    def eigen_decomposition(matrix):
        """Perform eigenvalue and eigenvector decomposition."""
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        return eigenvalues, eigenvectors

    # Matrix Decomposition
    @staticmethod
    def lu_decomposition(matrix):
        """Perform LU decomposition (Lower and Upper triangular decomposition)."""
        from scipy.linalg import lu
        P, L, U = lu(matrix)
        return P, L, U

    @staticmethod
    def qr_decomposition(matrix):
        """Perform QR decomposition (Orthogonal and Upper triangular matrices)."""
        Q, R = np.linalg.qr(matrix)
        return Q, R

    @staticmethod
    def svd(matrix):
        """Perform Singular Value Decomposition (SVD)."""
        U, S, V = np.linalg.svd(matrix)
        return U, S, V

    @staticmethod
    def cholesky(matrix):
        """Perform Cholesky decomposition (for positive-definite matrices)."""
        return np.linalg.cholesky(matrix)

    # Advanced Functions
    @staticmethod
    def matrix_exponential(matrix):
        """Compute the matrix exponential."""
        from scipy.linalg import expm
        return expm(matrix)

    @staticmethod
    def matrix_logarithm(matrix):
        """Compute the matrix logarithm."""
        from scipy.linalg import logm
        return logm(matrix)

    # Utility Functions
    @staticmethod
    def diagonal(matrix):
        """Extract the diagonal elements of the matrix."""
        return np.diag(matrix)

    @staticmethod
    def identity(size):
        """Create an identity matrix of the given size."""
        return np.eye(size)

    @staticmethod
    def zero_matrix(rows, cols):
        """Create a matrix filled with zeros."""
        return np.zeros((rows, cols))

    @staticmethod
    def outer_product(vector1, vector2):
        """Compute the outer product of two vectors."""
        return np.outer(vector1, vector2)

    @staticmethod
    def cross_product(vector1, vector2):
        """Compute the cross product of two vectors (for 3D vectors)."""
        return np.cross(vector1, vector2)

    @staticmethod
    def pad_matrix(matrix, pad_width, constant_values=0):
        """Pad the matrix with zeros or a specified constant value."""
        return np.pad(matrix, pad_width, constant_values=constant_values)

    @staticmethod
    def rotate_matrix_90(matrix):
        """Rotate the matrix 90 degrees counterclockwise."""
        return np.rot90(matrix)

# Exam
