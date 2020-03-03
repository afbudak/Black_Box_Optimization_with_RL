import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

mu1 = np.float64(30)
sigma1 = np.float64(10)

mu2 = np.float64(20)
sigma2 = np.float64(5)

mu_t = 0.5 * (mu1 + mu2)
sigma_t = 0.5 * np.hypot(sigma1, sigma2)

t1 = tfp.distributions.Normal(loc=mu1, scale=sigma1)
t2 = tfp.distributions.Normal(loc=mu2, scale=sigma2)
t = tfp.distributions.Normal(loc=mu_t, scale=sigma_t)

t10 = np.float64(10.)

p_r1_given_t10 = t1.prob(t10) / (t1.prob(t10) + t2.prob(t10))
p_r2_given_t10 = t2.prob(t10) / (t1.prob(t10) + t2.prob(t10))

print(f"p(r=r1 | t=10) = {p_r1_given_t10}")
print(f"p(r=r2 | t=10) = {p_r2_given_t10}")
