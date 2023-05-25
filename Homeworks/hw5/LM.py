import numpy as np


class LM:
    """
    Implementation of Levenberg–Marquardt algorithm
    """
    def __init__(self, initH, mkpts0, mkpts1, gamma: float = 0.001, max_iter=50,
                 mag_J_thr: int = 1e14, mag_delta_p_thr: float = 1e-14):
        self.H = initH  # 3x3
        self.gamma = gamma  # scales initial damping ratio mu
        self.mkpts0 = mkpts0  # Nx2
        self.mkpts1 = mkpts1  # Nx2
        self.max_iter = max_iter  # maximum iterations
        self.mag_J_thr = mag_J_thr
        self.mag_delta_p_thr = mag_delta_p_thr

        # check inputs
        assert self.H.shape == (3, 3), "Homography matrix should be a 3x3 matrix"
        assert 0 < self.gamma <= 1, "Gamma should be in (0, 1]"
        assert self.mkpts0.shape == self.mkpts1.shape, "Matching keypoints should have the same dimension"
        assert self.mkpts0.shape[1] == 2, "Matching keypoint should only contain x and y coordinates"
        assert self.mag_J_thr > 0, "Magnitude threshold of jacobian matrix should be positive"
        assert self.mag_delta_p_thr > 0, "Magnitude threshold of delta p vector should be positive"

        # flatten homography matrix to get parameter vector
        self.p = self.H.flatten()

        # flatten matching keypoints coordinates to get input and output vectors
        self.x = self.mkpts0.flatten().astype(float)  # coordinates vector in domain image
        self.x_prime = self.mkpts1.flatten().astype(float)  # coordinates vector in range image
        self.num = int(len(self.x) / 2)  # total number of correspondences
        print(f"Initial H: {self.H}")
        print(f"{self.num} matching keypoints")

    def f(self, p):
        """
        Calculate (re-)projection vector based on current parameter vector
        :param p:
        :return:
        """
        x_prime = np.zeros_like(self.x).astype(float)  # 1 x 2N vector

        den = self._denominator(p)  # 1 x N denominator vector
        num1 = self._numerator1(p)  # 1 x N numerator vector for function 1
        num2 = self._numerator2(p)  # 1 x N numerator vector for function 2

        x_prime[::2] = num1 / den
        x_prime[1::2] = num2 / den
        # print(x_prime)
        return x_prime

    def error(self, p):
        """
        (Re-)projection error vector
        :param p:
        :return: 1 x N
        """
        return self.x_prime - self.f(p)

    def cost(self, p):
        """
        Calculate squared norm of the (re-)projection error vector
        :param p:
        :return:
        """
        return np.sum((self.error(p)) ** 2)

    def pred_cost(self, p, delta_p, mu):
        """

        :param p:
        :param delta_p:
        :param mu:
        :return:
        """
        return delta_p @ self.J(p).T @ self.error(p) + mu * delta_p @ delta_p

    def J(self, p):
        """
        Calculate 2N x 9 jacobian matrix w.r.t parameter vector p
        :param p:
        :return:
        """
        j = np.zeros((self.num * 2, 9)).astype(float)  # 2N * 9 matrix

        # construct basic vector elements
        x_homog = self._x_homog()  # 3 x N matrix
        den = self._denominator(p)  # 1 x N vector
        num1 = self._numerator1(p)  # 1 x N numerator vector for function 1
        num2 = self._numerator2(p)  # 1 x N numerator vector for function 2

        # construct blocks for Jacobian matrix by vector elements
        block1 = (x_homog / den).T  # N x 3 matrix
        block2 = (-x_homog * num1 / den ** 2).T  # N x 3 * N x 1 / (N x 1)^2 = N x 3
        block3 = (-x_homog * num2 / den ** 2).T  # N x 3 * N x 1 / (N x 1)^2 = N x 3

        # check dimensions
        assert block1.shape == (self.num, 3)
        assert block2.shape == (self.num, 3)
        assert block3.shape == (self.num, 3)

        # fill in blocks
        j[::2, :3] = block1
        j[1::2, 3:6] = block1
        j[::2, 6:] = block2
        j[1::2, 6:] = block3

        return j

    def JtJ(self, p):
        """
        Jacobian's transpose times Jacobian matrix, 9 x 9
        :param p:
        :return: 9 x 9 matrix
        """
        j = self.J(p)
        jtj = j.T @ j
        assert jtj.shape == (9, 9)
        return jtj

    def _x_homog(self):
        """
        Domain points in homogeneous coordinates
        :return: 3 x N
        """
        x_homog = np.vstack((self.x.reshape((-1, 2)).T, [1] * self.num))
        assert x_homog.shape == (3, self.num)
        return x_homog

    def _denominator(self, p):
        """
        1 x N denominator vector
        :param p:
        :return: 1 x N vector
        """
        den = p[6:] @ self._x_homog()
        assert den.shape == (self.num,)
        return den

    def _numerator1(self, p):
        """
        1 x N numerator vector for function 1
        :param p:
        :return:
        """
        num1 = p[:3] @ self._x_homog()
        assert num1.shape == (self.num,)
        return num1

    def _numerator2(self, p):
        """
        1 x N numerator vector for function 2
        :param p:
        :return:
        """
        num2 = p[3:6] @ self._x_homog()
        assert num2.shape == (self.num,)
        return num2

    def refine(self, get_cost=True):
        """
        Main function that refines the given homography matrix iteratively to get to sub-pixel re-projection error
        :return: refined 3 x 3 homography matrix
        """
        print(f"Start LM algorithm...")

        mu = max(np.diag(self.JtJ(self.p))) * self.gamma
        print(f"Initial mu is {mu}")

        cost_list = []

        for i in range(self.max_iter):
            # get delta_p in augmented normal equation
            j = self.J(self.p)
            jt_j = j.T @ j
            jt_j_norm = np.linalg.norm(jt_j)
            print(f"JtJ norm is {jt_j_norm}")

            # stop condition 1: too small Jacobian norm
            if jt_j_norm > self.mag_J_thr:
                print(f"Too large JtJ norm to continue, quit!")
                break

            delta_p = np.linalg.pinv(jt_j + np.diag([mu] * 9)) @ j.T @ self.error(self.p)

            # stop condition 2: too small delta_p norm
            delta_p_norm = np.linalg.norm(delta_p)
            print(f"Iteration {i}: delta_p = {delta_p}, delta_p_norm = {delta_p_norm}")
            if delta_p_norm < self.mag_delta_p_thr:
                print(f"Too small delta_p norm to continue, quit!")
                break

            new_p = self.p + delta_p

            # get all costs
            cur_cost = self.cost(self.p)

            if get_cost:
                cost_list.append(cur_cost)

            next_cost = self.cost(new_p)
            pred_cost = self.pred_cost(self.p, delta_p, mu)
            cost_diff = cur_cost - next_cost  # want this to be positive
            ratio = cost_diff / pred_cost  # want this to be positive also
            print(f"Current cost: {cur_cost}, next cost: {next_cost}, cost_diff: {cost_diff}, ratio: {ratio}")

            # update damping
            mu = mu * max(1/3, 1 - (2 * ratio - 1) ** 3)
            print(f"Updated mu is {mu}")

            # update parameter vector only when cost decreases
            if ratio > 0:
                self.p = new_p
                print(f"Update parameter vector!")
            else:
                print(f"Cost increased, only update mu!")

        print(f"LM finished!")
        return self.p.reshape((3, 3)), cost_list





