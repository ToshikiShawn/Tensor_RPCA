import numpy as np
from matplotlib import pylab as plt
from numpy.linalg import svd
from PIL import Image

class TRPCA:

    def converged(self, L, E, X, L_new, E_new):
        '''
        judge convered or not
        '''
        eps = 1e-8
        condition1 = np.max(L_new - L) < eps
        condition2 = np.max(E_new - E) < eps
        condition3 = np.max(L_new + E_new - X) < eps
        return condition1 and condition2 and condition3

    def SoftShrink(self, X, tau):
        '''
        apply soft thesholding
        '''
        z = np.sign(X) * (abs(X) - tau) * ((abs(X) - tau) > 0)

        return z

    def SVDShrink(self, X, tau):
        '''
        apply tensor-SVD and soft thresholding
        '''
        W_bar = np.empty((X.shape[0], X.shape[1], 0), complex)
        D = np.fft.fft(X)
        for i in range (3):
            if i < 3:
                U, S, V = svd(D[:, :, i], full_matrices = False)
                S = self.SoftShrink(S, tau)
                S = np.diag(S)
                w = np.dot(np.dot(U, S), V)
                W_bar = np.append(W_bar, w.reshape(X.shape[0], X.shape[1], 1), axis = 2)
            if i == 3:
                W_bar = np.append(W_bar, (w.conjugate()).reshape(X.shape[0], X.shape[1], 1))
        return np.fft.ifft(W_bar).real


    def ADMM(self, X):
        '''
        Solve
        min (nuclear_norm(L)+lambda*l1norm(E)), subject to X = L+E
        L,E
        by ADMM
        '''
        m, n, l = X.shape
        rho = 1.1
        mu = 1e-3
        mu_max = 1e10
        max_iters = 1000
        lamb = (max(m, n) * l) ** -0.5
        L = np.zeros((m, n, l), float)
        E = np.zeros((m, n, l), float)
        Y = np.zeros((m, n, l), float)
        iters = 0
        while True:
            iters += 1
            # update L(recovered image)
            L_new = self.SVDShrink(X - E + (1/mu) * Y, 1/mu)
            # update E(noise)
            E_new = self.SoftShrink(X - L_new + (1/mu) * Y, lamb/mu)
            Y += mu * (X - L_new - E_new)
            mu = min(rho * mu, mu_max)
            if self.converged(L, E, X, L_new, E_new) or iters >= max_iters:
                return L_new, E_new
            else:
                L, E = L_new, E_new
                print(np.max(X - L - E))


# Load Data
X = np.array(Image.open(r'photo.jpg'))

# add noise(make some pixels black at the rate of 10%)
k = np.random.rand(X.shape[0], X.shape[1]) > 0.1
K = np.empty((X.shape[0], X.shape[1], 0), np.uint8)
for i in range (3):
    K = np.append(K, k.reshape(X.shape[0], X.shape[1], 1), axis = 2)
X_bar = X * K

# image denoising
TRPCA = TRPCA()
L, E = TRPCA.ADMM(X)
L = np.array(L).astype(np.uint8)
E = np.array(E).astype(np.uint8)
plt.subplot(131)
plt.imshow(X)
plt.title('original image')
plt.subplot(132)
plt.imshow(X_bar)
plt.title('image with noise')
plt.subplot(133)
plt.imshow(L)
plt.title('recovered image')
plt.show()
